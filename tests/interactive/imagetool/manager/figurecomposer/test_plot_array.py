import types
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
from qtpy import QtCore, QtWidgets

import erlab.interactive._figurecomposer._rendering as figurecomposer_rendering
import erlab.interactive._figurecomposer._text as figurecomposer_text
import erlab.interactive._figurecomposer._ui._editor_controls as _editor_controls
import erlab.interactive._stylesheets
import erlab.plotting as eplt
from erlab.interactive._figurecomposer import (
    FigureAxesSelectionState,
    FigureComposerTool,
    FigureDataSelectionState,
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
from erlab.interactive._figurecomposer._model import (
    _operation_metadata as figurecomposer_operation_metadata,
)
from erlab.interactive._figurecomposer._operations import (
    _plot_array as figurecomposer_plot_array,
)
from tests.interactive.imagetool.manager.helpers import _exec_generated_code

from ._common import (
    _activate_combo_index,
    _activate_combo_text,
    _figure_composer_image_source,
    _select_operation_rows,
    _set_figure_stylesheets,
)


def _source_selection_widget(
    tool: FigureComposerTool,
    widget_type: type[QtWidgets.QWidget],
    dim: str,
    *,
    field: str | None = None,
) -> QtWidgets.QWidget:
    widget = next(
        (
            candidate
            for candidate in tool.source_panel.source_selection_controls.findChildren(
                widget_type
            )
            if not candidate.signalsBlocked()
            if candidate.property("figure_composer_source_selection_dim") == dim
            if field is None
            or candidate.property("figure_composer_source_selection_field") == field
        ),
        None,
    )
    assert widget is not None
    return widget


def _source_selection_edit(
    tool: FigureComposerTool, dim: str, field: str
) -> QtWidgets.QLineEdit:
    return typing.cast(
        "QtWidgets.QLineEdit",
        _source_selection_widget(tool, QtWidgets.QLineEdit, dim, field=field),
    )


def _activate_source_selection_mode(
    tool: FigureComposerTool, dim: str, mode: str
) -> None:
    combo = typing.cast(
        "QtWidgets.QComboBox",
        _source_selection_widget(tool, QtWidgets.QComboBox, dim),
    )
    index = combo.findData(mode)
    assert index >= 0
    combo.setCurrentIndex(index)
    combo.activated.emit(index)


def test_figure_composer_plot_array_source_selector_clears_legacy_selection(
    qtbot,
) -> None:
    first = _figure_composer_image_source("first")
    second = _figure_composer_image_source("second")
    operation = FigureOperationState.plot_array(
        label="plot_array",
        source="first_source",
        map_selections=(
            FigureDataSelectionState(source="first_source", qsel={"eV": 0.0}),
        ),
    )
    tool = FigureComposerTool.from_sources(
        {"first_source": first, "second_source": second},
        sources=(
            FigureSourceState(name="first_source", label="first"),
            FigureSourceState(name="second_source", label="second"),
        ),
        operations=(operation,),
        primary_source="first_source",
    )
    qtbot.addWidget(tool)
    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(0)
    )
    tool.operation_editor.select_section("sources")

    combo = next(
        (
            candidate
            for candidate in tool.findChildren(
                QtWidgets.QComboBox, "figureComposerPlotArraySourceCombo"
            )
            if tool.operation_editor.control_signal_allowed(candidate)
        ),
        None,
    )
    assert combo is not None
    index = combo.findData("second_source")
    assert index >= 0
    combo.setCurrentIndex(index)
    combo.activated.emit(index)

    updated = tool.tool_status.operations[0]
    assert updated.sources == ("second_source",)
    assert updated.map_selections == ()


def test_figure_composer_plot_array_source_code_handles_missing_primary(
    qtbot, monkeypatch
) -> None:
    data = xr.DataArray(np.arange(2.0), dims=("x",), name="data")
    tool = FigureComposerTool.from_sources(
        {"data": data},
        sources=(FigureSourceState(name="data"),),
        operations=(FigureOperationState.plot_array(label="array", source="data"),),
        primary_source="data",
    )
    qtbot.addWidget(tool)
    operation = tool.tool_status.operations[0].model_copy(update={"sources": ()})
    monkeypatch.setattr(
        figurecomposer_plot_array,
        "_selected_plot_array_data",
        lambda *_args, **_kwargs: data,
    )

    assert figurecomposer_plot_array._plot_array_source_code(tool, operation) is None


def test_figure_composer_source_selection_editor_updates_source_snapshot(
    qtbot,
) -> None:
    data = xr.DataArray(
        np.arange(24.0).reshape(2, 3, 2, 2),
        dims=("hv", "eV", "beta", "alpha"),
        coords={
            "hv": [30.0, 40.0],
            "eV": [-1.0, 0.0, 1.0],
            "beta": [-0.5, 0.5],
            "alpha": [0.0, 1.0],
        },
        name="map",
    )
    operation = FigureOperationState.plot_array(label="plot_array", source="data")
    tool = FigureComposerTool.from_sources(
        {"data": data},
        sources=(FigureSourceState(name="data", label="data"),),
        operations=(operation,),
        primary_source="data",
    )
    qtbot.addWidget(tool)
    tool.editor_tabs.setCurrentWidget(tool.source_panel)

    eV_mode = typing.cast(
        "QtWidgets.QComboBox",
        _source_selection_widget(tool, QtWidgets.QComboBox, "eV"),
    )
    assert eV_mode.currentData() == "keep"
    _activate_source_selection_mode(tool, "eV", "qsel")
    qsel_edit = _source_selection_edit(tool, "eV", "value")
    qsel_edit.setText("1.0")
    qsel_edit.editingFinished.emit()

    [source] = tool.source_states()
    assert source.qsel == {"eV": 1.0}
    assert source.isel == {}
    assert source.mean_dims == ()
    assert source.selection_source is None
    xr.testing.assert_identical(tool.source_data()["data"], data.qsel(eV=1.0))
    assert tool.tool_status.operations[0].map_selections == ()

    _activate_source_selection_mode(tool, "eV", "isel")
    isel_eV_edit = _source_selection_edit(tool, "eV", "value")
    isel_eV_edit.setText("1")
    isel_eV_edit.editingFinished.emit()
    [source] = tool.source_states()
    assert source.isel == {"eV": 1}
    assert source.qsel == {}

    _activate_source_selection_mode(tool, "hv", "isel")
    isel_edit = _source_selection_edit(tool, "hv", "value")
    isel_edit.setText("1")
    isel_edit.editingFinished.emit()

    [source] = tool.source_states()
    assert source.isel == {"eV": 1, "hv": 1}
    xr.testing.assert_identical(
        tool.source_data()["data"],
        data.isel(eV=1, hv=1),
    )


def test_figure_composer_source_qsel_to_keep_clears_selection(qtbot) -> None:
    data = xr.DataArray(
        np.arange(12.0).reshape(3, 2, 2),
        dims=("eV", "beta", "alpha"),
        coords={
            "eV": [-1.0, 0.0, 1.0],
            "beta": [-0.5, 0.5],
            "alpha": [0.0, 1.0],
        },
        name="map",
    )
    operation = FigureOperationState.plot_array(label="plot_array", source="data")
    tool = FigureComposerTool.from_sources(
        {"data": data},
        sources=(FigureSourceState(name="data", label="data"),),
        operations=(operation,),
        primary_source="data",
    )
    qtbot.addWidget(tool)
    tool.editor_tabs.setCurrentWidget(tool.source_panel)

    _activate_source_selection_mode(tool, "eV", "qsel")
    qsel_edit = _source_selection_edit(tool, "eV", "value")
    assert qsel_edit.isEnabled()
    assert not _source_selection_edit(tool, "eV", "value").isHidden()
    assert not _source_selection_edit(tool, "eV", "width").isHidden()
    qsel_edit.setText("0.0")
    qsel_edit.editingFinished.emit()

    [source] = tool.source_states()
    assert source.qsel == {"eV": 0.0}
    xr.testing.assert_identical(tool.source_data()["data"], data.qsel(eV=0.0))

    _activate_source_selection_mode(tool, "eV", "keep")

    [source] = tool.source_states()
    assert source.isel == {}
    assert source.qsel == {}
    assert source.mean_dims == ()
    assert source.selection_source is None
    xr.testing.assert_identical(tool.source_data()["data"], data)
    value_edit = _source_selection_edit(tool, "eV", "value")
    width_edit = _source_selection_edit(tool, "eV", "width")
    assert value_edit.isHidden()
    assert width_edit.isHidden()
    assert value_edit.text() == ""


def test_figure_composer_source_keep_clears_stale_qsel_input_error(
    qtbot,
) -> None:
    data = xr.DataArray(
        np.arange(12.0).reshape(3, 2, 2),
        dims=("eV", "beta", "alpha"),
        coords={
            "eV": [-1.0, 0.0, 1.0],
            "beta": [-0.5, 0.5],
            "alpha": [0.0, 1.0],
        },
        name="map",
    )
    operation = FigureOperationState.plot_array(label="plot_array", source="data")
    tool = FigureComposerTool.from_sources(
        {"data": data},
        sources=(FigureSourceState(name="data", label="data"),),
        operations=(operation,),
        primary_source="data",
    )
    qtbot.addWidget(tool)
    tool.editor_tabs.setCurrentWidget(tool.source_panel)

    _activate_source_selection_mode(tool, "eV", "qsel")
    qsel_edit = _source_selection_edit(tool, "eV", "value")
    qsel_edit.setText("slice(-0.5, 0.5)")
    qsel_edit.editingFinished.emit()

    [source] = tool.source_states()
    assert source.qsel == {"eV": slice(-0.5, 0.5)}

    qsel_edit = _source_selection_edit(tool, "eV", "value")
    qsel_edit.setText("")
    qsel_edit.editingFinished.emit()
    assert not tool.source_panel.source_validation_label.isHidden()

    _activate_source_selection_mode(tool, "eV", "keep")

    [source] = tool.source_states()
    assert source.qsel == {}
    assert tool.source_panel.source_validation_label.isHidden()


def test_figure_composer_source_qsel_width_editor_updates_selection(
    qtbot,
) -> None:
    data = _figure_composer_image_source("data")
    operation = FigureOperationState.plot_array(
        label="plot_array",
        source="data",
        map_selections=(
            FigureDataSelectionState(
                source="data",
                qsel={"eV": 0.0, "eV_width": 0.2},
            ),
        ),
    )
    tool = FigureComposerTool.from_sources(
        {"data": data},
        sources=(FigureSourceState(name="data", label="data"),),
        operations=(operation,),
        primary_source="data",
    )
    qtbot.addWidget(tool)

    normalized_operation = tool.tool_status.operations[0]
    assert normalized_operation.sources == ("data_selected",)
    assert normalized_operation.map_selections == ()
    [raw_source, selected_source] = tool.source_states()
    assert raw_source.name == "data"
    assert selected_source.name == "data_selected"
    assert selected_source.qsel == {"eV": 0.0, "eV_width": 0.2}
    assert selected_source.selection_source == "data"
    selected = figurecomposer_plot_array._safe_selected_plot_array_data(
        tool._document, normalized_operation
    )
    assert selected is not None
    xr.testing.assert_identical(selected, data.qsel(eV=0.0, eV_width=0.2))
    assert "data_selected = data.qsel(eV=0.0, eV_width=0.2)" in tool.generated_code()

    tool.editor_tabs.setCurrentWidget(tool.source_panel)
    tool.source_panel.set_selected_names(
        ("data_selected",), current_name="data_selected"
    )
    tool._refresh_source_selection_editor()
    mode_combo = typing.cast(
        "QtWidgets.QComboBox",
        _source_selection_widget(tool, QtWidgets.QComboBox, "eV"),
    )
    value_edit = _source_selection_edit(tool, "eV", "value")
    width_edit = _source_selection_edit(tool, "eV", "width")
    assert not value_edit.isHidden()
    assert not width_edit.isHidden()
    assert value_edit.text() == "0.0"
    assert width_edit.text() == "0.2"
    assert mode_combo.toolTip()
    assert value_edit.toolTip()
    assert width_edit.toolTip()
    assert len({mode_combo.toolTip(), value_edit.toolTip(), width_edit.toolTip()}) == 3

    width_edit.setText("0.5")
    width_edit.editingFinished.emit()
    selected_source = tool.source_states()[1]
    assert selected_source.qsel == {"eV": 0.0, "eV_width": 0.5}
    xr.testing.assert_identical(
        tool.source_data()["data_selected"],
        data.qsel(eV=0.0, eV_width=0.5),
    )

    _activate_source_selection_mode(tool, "eV", "isel")
    value_edit = _source_selection_edit(tool, "eV", "value")
    value_edit.setText("1")
    value_edit.editingFinished.emit()
    selected_source = tool.source_states()[1]
    assert selected_source.isel == {"eV": 1}
    assert not _source_selection_edit(tool, "eV", "value").isHidden()
    assert _source_selection_edit(tool, "eV", "width").isHidden()

    _activate_source_selection_mode(tool, "eV", "qsel")
    value_edit = _source_selection_edit(tool, "eV", "value")
    value_edit.setText("0.0")
    value_edit.editingFinished.emit()
    width_edit = _source_selection_edit(tool, "eV", "width")
    width_edit.setText("0.1")
    width_edit.editingFinished.emit()
    selected_source = tool.source_states()[1]
    assert selected_source.qsel == {"eV": 0.0, "eV_width": 0.1}

    _activate_source_selection_mode(tool, "eV", "mean")
    selected_source = tool.source_states()[1]
    assert selected_source.mean_dims == ("eV",)
    assert _source_selection_edit(tool, "eV", "value").isHidden()
    assert _source_selection_edit(tool, "eV", "width").isHidden()


def test_figure_composer_plot_array_add_operation_promotes_selection_to_source(
    qtbot,
) -> None:
    first = _figure_composer_image_source("data_3")
    second = _figure_composer_image_source("data_4")
    initial_operation = FigureOperationState.plot_array(
        label="plot_array",
        source="data_3",
        map_selections=(FigureDataSelectionState(source="data_3", qsel={"eV": 0.0}),),
    )
    tool = FigureComposerTool.from_sources(
        {"data_3": first},
        sources=(FigureSourceState(name="data_3", label="data_3"),),
        operations=(initial_operation,),
        primary_source="data_3",
    )
    qtbot.addWidget(tool)
    tool.add_sources(
        (FigureSourceState(name="data_4", label="data_4"),),
        {"data_4": second},
    )
    data_changed: list[None] = []
    tool.sigDataChanged.connect(lambda: data_changed.append(None))
    tool.add_operation(
        FigureOperationState.plot_array(
            label="plot_array",
            source="data_4",
            map_selections=(
                FigureDataSelectionState(source="data_4", qsel={"eV": 0.5}),
            ),
        )
    )

    data_4_source = next(
        source for source in tool.source_states() if source.name == "data_4_selected"
    )
    assert data_4_source.selection_source == "data_4"
    assert data_4_source.qsel == {"eV": 0.5}
    assert tool.tool_status.operations[-1].sources == ("data_4_selected",)
    assert tool.tool_status.operations[-1].map_selections == ()
    assert data_changed == [None]
    xr.testing.assert_identical(
        tool.source_data()["data_4_selected"], second.qsel(eV=0.5)
    )


def test_figure_composer_restored_selected_source_applies_source_selection(
    qtbot,
) -> None:
    data = _figure_composer_image_source("data_3")
    selected_source = FigureSourceState(
        name="data_3_selected",
        label="data_3 selection",
        selection_source="data_3",
        qsel={"eV": 0.0},
    )
    tool = FigureComposerTool.from_sources(
        {"data_3": data, "data_3_selected": data.qsel(eV=0.0)},
        sources=(
            FigureSourceState(name="data_3", label="data_3"),
            selected_source,
        ),
        operations=(
            FigureOperationState.plot_array(
                label="plot_array",
                source="data_3_selected",
            ),
        ),
        primary_source="data_3",
    )
    qtbot.addWidget(tool)

    tool._restore_persistence_data_items(
        {"data_3_selected": data},
        xr.Dataset(),
    )

    xr.testing.assert_identical(
        tool.source_data()["data_3_selected"], data.qsel(eV=0.0)
    )
    xr.testing.assert_identical(
        tool._document.source_selection_base_data["data_3_selected"], data
    )
    assert tool.tool_status.operations[0].sources == ("data_3_selected",)
    assert tool.tool_status.operations[0].map_selections == ()


def test_figure_composer_plot_array_selection_error_is_visible(qtbot) -> None:
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("y", "x"),
        coords={"y": [0.0, 1.0], "x": [0.0, 1.0]},
        name="map",
    )
    operation = FigureOperationState.plot_array(
        label="plot_array",
        source="data",
        map_selections=(
            FigureDataSelectionState(source="data", qsel={"missing": 0.0}),
        ),
    )
    tool = FigureComposerTool.from_sources(
        {"data": data},
        sources=(FigureSourceState(name="data", label="data"),),
        operations=(operation,),
        primary_source="data",
    )
    qtbot.addWidget(tool)

    normalized_operation = tool.tool_status.operations[0]
    assert normalized_operation.map_selections == ()
    assert "missing" in figurecomposer_plot_array._display_text(
        tool, normalized_operation
    )

    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(0)
    )
    assert "selection" not in tool.operation_editor.section_keys


def test_figure_composer_plot_array_has_no_operation_selection_page(qtbot) -> None:
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("y", "x"),
        coords={"y": [0.0, 1.0], "x": [0.0, 1.0]},
        name="map",
    )
    tool = FigureComposerTool.from_sources(
        {"data": data},
        sources=(FigureSourceState(name="data", label="data"),),
        operations=(
            FigureOperationState.plot_array(label="missing", source="missing"),
        ),
        primary_source="data",
    )
    qtbot.addWidget(tool)
    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(0)
    )
    assert tool.operation_editor.section_keys == (
        "sources",
        "axes",
        "view",
        "colors",
        "advanced",
    )

    mixed_tool = FigureComposerTool.from_sources(
        {
            "first": data,
            "second": data,
        },
        sources=(
            FigureSourceState(name="first", label="first"),
            FigureSourceState(name="second", label="second"),
        ),
        operations=(
            FigureOperationState.plot_array(label="first", source="first"),
            FigureOperationState.plot_array(label="second", source="second"),
        ),
        primary_source="first",
    )
    qtbot.addWidget(mixed_tool)
    _select_operation_rows(mixed_tool, (0, 1))
    assert mixed_tool.operation_editor.section_keys == (
        "sources",
        "axes",
        "view",
        "colors",
        "advanced",
    )


def test_figure_composer_plot_array_colormap_activation_updates_recipe(
    qtbot,
) -> None:
    data = _figure_composer_image_source("data").isel(eV=0)
    operation = FigureOperationState.plot_array(label="plot_array", source="data")
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
                "figureComposerPlotArrayCmapCombo",
            )
            if tool.operation_editor.control_signal_allowed(candidate)
        ),
        None,
    )
    assert cmap_combo is not None
    cmap_combo.load_all()
    assert tool.tool_status.operations[0].cmap is None
    current_cmap = cmap_combo.currentText()
    target_index = next(
        index
        for index in range(cmap_combo.count())
        if cmap_combo.itemText(index) != current_cmap
    )
    cmap = cmap_combo.itemText(target_index)
    cmap_combo.setCurrentText(cmap)
    cmap_combo.activated.emit(cmap_combo.currentIndex())

    assert tool.tool_status.operations[0].cmap == cmap


def test_figure_composer_plot_array_colorcet_selection_uses_matplotlib_name(
    qtbot,
) -> None:
    pytest.importorskip("colorcet")

    data = _figure_composer_image_source("data").isel(eV=0)
    operation = FigureOperationState.plot_array(label="plot_array", source="data")
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
                "figureComposerPlotArrayCmapCombo",
            )
            if tool.operation_editor.control_signal_allowed(candidate)
        ),
        None,
    )
    assert cmap_combo is not None
    cmap_combo.load_all()
    cmap_combo.setCurrentText("CET_C1")
    cmap_combo.activated.emit(cmap_combo.currentIndex())

    operation = tool.tool_status.operations[0]
    assert operation.cmap == "cet_CET_C1"
    assert (
        figurecomposer_plot_array._plot_array_kwargs(operation)["cmap"] == "cet_CET_C1"
    )
    assert (
        figurecomposer_plot_array._plot_array_code_kwargs(operation)["cmap"]
        == "cet_CET_C1"
    )

    figurecomposer_rendering._render_into_figure(tool, tool.figure, sync_visible=False)

    assert tool.figure.axes[0].images[-1].get_cmap().name == "cet_CET_C1"


def test_figure_composer_plot_array_default_colormap_editor_uses_stylesheet(
    qtbot,
    tmp_path: Path,
) -> None:
    style_name = "erlab-test-image-cmap"
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
        operation = FigureOperationState.plot_array(label="plot_array", source="data")
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
                    "figureComposerPlotArrayCmapCombo",
                )
                if tool.operation_editor.control_signal_allowed(candidate)
            ),
            None,
        )
        cmap_reverse_check = next(
            (
                candidate
                for candidate in tool.findChildren(
                    QtWidgets.QCheckBox, "figureComposerPlotArrayCmapReverseCheck"
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

        figurecomposer_rendering._render_into_figure(
            tool, tool.figure, sync_visible=False
        )
        assert tool.figure.axes[0].images[-1].get_cmap().name == "plasma"
        assert "cmap" not in figurecomposer_plot_array._plot_array_kwargs(
            tool.tool_status.operations[0]
        )

        cmap_reverse_check.setChecked(True)

        assert tool.tool_status.operations[0].cmap == "plasma_r"
    finally:
        mpl_style_core.USER_LIBRARY_PATHS.remove(str(style_dir))
        mpl_style.reload_library()


def test_figure_composer_plot_array_colormap_editor_initialization_edges(
    qtbot,
) -> None:
    data = _figure_composer_image_source("data").isel(eV=0)
    explicit = FigureOperationState.plot_array(
        label="explicit",
        source="data",
    ).model_copy(update={"cmap": "magma"})
    tool = FigureComposerTool.from_sources(
        {"data": data},
        sources=(FigureSourceState(name="data", label="data"),),
        operations=(explicit,),
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
                "figureComposerPlotArrayCmapCombo",
            )
            if tool.operation_editor.control_signal_allowed(candidate)
        ),
        None,
    )
    assert cmap_combo is not None
    assert cmap_combo.currentData() == "magma"

    missing_cmap_tool = FigureComposerTool.from_sources(
        {"data": data},
        sources=(FigureSourceState(name="data", label="data"),),
        operations=(explicit.model_copy(update={"cmap": "missing_colormap_name"}),),
        primary_source="data",
    )
    qtbot.addWidget(missing_cmap_tool)
    missing_cmap_tool.operation_panel.operation_list.setCurrentItem(
        missing_cmap_tool.operation_panel.operation_list.topLevelItem(0)
    )
    missing_cmap_tool.operation_editor.select_section("colors")
    assert (
        missing_cmap_tool.findChild(
            erlab.interactive.colors.ColorMapComboBox,
            "figureComposerPlotArrayCmapCombo",
        )
        is not None
    )

    mixed_tool = FigureComposerTool.from_sources(
        {"data": data},
        sources=(FigureSourceState(name="data", label="data"),),
        operations=(
            explicit,
            FigureOperationState.plot_array(label="viridis", source="data").model_copy(
                update={"cmap": "viridis"}
            ),
        ),
        primary_source="data",
    )
    qtbot.addWidget(mixed_tool)
    _select_operation_rows(mixed_tool, (0, 1))
    mixed_tool.operation_editor.select_section("colors")
    mixed_cmap_combo = next(
        (
            candidate
            for candidate in mixed_tool.findChildren(
                erlab.interactive.colors.ColorMapComboBox,
                "figureComposerPlotArrayCmapCombo",
            )
            if mixed_tool.operation_editor.control_signal_allowed(candidate)
        ),
        None,
    )
    assert mixed_cmap_combo is not None
    assert mixed_cmap_combo.currentData() is _editor_controls.MIXED_VALUE


def test_figure_composer_plot_array_add_action_and_plain_2d_codegen(
    qtbot, monkeypatch
) -> None:
    data = _figure_composer_image_source("data").isel(eV=0)
    tool = FigureComposerTool.from_sources(
        {"data": data},
        sources=(FigureSourceState(name="data", label="data"),),
        operations=(),
        primary_source="data",
    )
    qtbot.addWidget(tool)

    action = next(
        action
        for action in tool.operation_panel.add_step_menu.actions()
        if action.data() == FigureOperationKind.PLOT_ARRAY.value
    )
    action.trigger()

    operation = tool.tool_status.operations[-1]
    assert operation.kind == FigureOperationKind.PLOT_ARRAY
    assert operation.sources == ("data",)
    assert operation.map_selections == ()

    captured: list[xr.DataArray] = []

    def capture_plot_array(arr, **_kwargs):
        captured.append(arr)

    monkeypatch.setattr(eplt, "plot_array", capture_plot_array)
    code = tool.generated_code()
    assert "eplt.plot_array(data" in code
    namespace = _exec_generated_code(code, {"data": data})
    assert "fig" in namespace
    xr.testing.assert_identical(captured[0], data)


def test_figure_composer_default_plot_array_uses_one_axis_in_multi_axis_setup(
    qtbot, monkeypatch
) -> None:
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("x", "y"),
        coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
        name="map",
    )
    tool = FigureComposerTool.from_sources(
        {"data": data},
        sources=(FigureSourceState(name="data", label="map"),),
        setup=FigureSubplotsState(nrows=1, ncols=2),
        primary_source="data",
    )
    qtbot.addWidget(tool)

    operation = tool.tool_status.operations[0]
    assert operation.kind == FigureOperationKind.PLOT_ARRAY
    assert operation.axes.axes == ((0, 0),)
    assert not figurecomposer_plot_array._has_invalid_target(tool._document, operation)

    captured: list[tuple[xr.DataArray, dict[str, typing.Any]]] = []

    def capture_plot_array(arr, **kwargs):
        captured.append((arr, kwargs))

    monkeypatch.setattr(eplt, "plot_array", capture_plot_array)
    namespace = _exec_generated_code(tool.generated_code(), {"data": data})

    assert "fig" in namespace
    xr.testing.assert_identical(captured[0][0], data)
    assert captured[0][1]["ax"] is namespace["axs"][0, 0]


def test_figure_composer_plot_array_render_and_generated_code(
    qtbot, monkeypatch
) -> None:
    data = _figure_composer_image_source("data")
    operation = FigureOperationState.plot_array(
        label="plot_array",
        source="data",
        map_selections=(FigureDataSelectionState(source="data", qsel={"eV": 0.0}),),
    ).model_copy(
        update={
            "transpose": True,
            "xlim": (-0.5, 0.5),
            "ylim": (-1.0, 1.0),
            "colorbar": "right",
            "cmap": "magma",
            "norm_gamma": 0.5,
            "vmin": 0.0,
            "vmax": 2.0,
            "aspect": "equal",
        }
    )
    tool = FigureComposerTool.from_sources(
        {"data": data},
        sources=(FigureSourceState(name="data", label="data"),),
        operations=(operation,),
        primary_source="data",
    )
    qtbot.addWidget(tool)
    captured: list[tuple[xr.DataArray, dict[str, typing.Any]]] = []
    preparation_calls: list[None] = []
    original_selected_data = (
        figurecomposer_plot_array._selected_plot_array_data_from_source
    )

    def capture_plot_array(arr, **kwargs):
        captured.append((arr, kwargs))

    def counted_selected_data(*args, **kwargs):
        preparation_calls.append(None)
        return original_selected_data(*args, **kwargs)

    monkeypatch.setattr(eplt, "plot_array", capture_plot_array)
    monkeypatch.setattr(
        figurecomposer_plot_array,
        "_selected_plot_array_data_from_source",
        counted_selected_data,
    )
    figurecomposer_rendering._render_into_figure(tool, tool.figure, sync_visible=False)

    assert len(captured) == 1
    rendered, kwargs = captured[0]
    assert rendered.dims == ("alpha", "beta")
    assert kwargs["ax"] is tool.figure.axes[0]
    assert kwargs["xlim"] == (-0.5, 0.5)
    assert kwargs["ylim"] == (-1.0, 1.0)
    assert kwargs["colorbar"] is True
    assert kwargs["cmap"] == "magma"
    assert kwargs["gamma"] == 0.5
    assert kwargs["vmin"] == 0.0
    assert kwargs["vmax"] == 2.0
    assert kwargs["aspect"] == "equal"

    code = tool.generated_code()
    assert "data_selected = data.qsel(eV=0.0)" in code
    assert "eplt.plot_array(data_selected.T" in code
    namespace = _exec_generated_code(code, {"data": data})
    assert "fig" in namespace

    preparation_calls.clear()
    figurecomposer_rendering._render_into_figure(tool, tool.figure, sync_visible=False)
    styled = tool.tool_status.operations[0].model_copy(update={"cmap": "viridis"})
    tool._document.replace_operation(0, styled)
    figurecomposer_rendering._render_into_figure(tool, tool.figure, sync_visible=False)
    assert preparation_calls == []

    tool._document.replace_source_payloads({"data": data + 1.0}, {})
    figurecomposer_rendering._render_into_figure(tool, tool.figure, sync_visible=False)
    assert preparation_calls == [None]


def test_figure_composer_plot_array_caches_conversion_aware_crop(
    qtbot, monkeypatch
) -> None:
    data = xr.DataArray(
        np.arange(63.0).reshape(7, 9),
        dims=("beta", "alpha"),
        coords={
            "beta": np.deg2rad(np.linspace(-60.0, 60.0, 7)),
            "alpha": np.deg2rad(np.linspace(-80.0, 80.0, 9)),
        },
        name="data",
    )
    operation = FigureOperationState.plot_array(
        label="plot_array", source="data"
    ).model_copy(
        update={
            "crop": True,
            "xlim": (-30.0, 30.0),
            "ylim": (-30.0, 30.0),
            "extra_kwargs": {"rad2deg": True},
        }
    )
    tool = FigureComposerTool.from_sources(
        {"data": data},
        sources=(FigureSourceState(name="data", label="data"),),
        operations=(operation,),
        primary_source="data",
    )
    qtbot.addWidget(tool)

    preparation_calls: list[bool] = []
    original_prepare = figurecomposer_plot_array._prepare_plot_array_data

    def counted_prepare(*args, **kwargs):
        preparation_calls.append(kwargs["rad2deg"])
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(
        figurecomposer_plot_array, "_prepare_plot_array_data", counted_prepare
    )

    figurecomposer_rendering._render_into_figure(tool, tool.figure, sync_visible=False)

    assert preparation_calls == [True]
    assert tool.figure.axes[0].images[0].get_array().shape == (3, 3)

    figurecomposer_rendering._render_into_figure(tool, tool.figure, sync_visible=False)

    assert preparation_calls == [True]

    updated = tool.tool_status.operations[0].model_copy(
        update={"extra_kwargs": {"rad2deg": False}}
    )
    tool._document.replace_operation(0, updated)
    figurecomposer_rendering._render_into_figure(tool, tool.figure, sync_visible=False)

    assert preparation_calls == [True, False]
    assert tool.figure.axes[0].images[0].get_array().shape == (7, 9)


def test_figure_composer_plot_array_aspect_control_updates_recipe(qtbot) -> None:
    data = _figure_composer_image_source("data").isel(eV=0)
    operation = FigureOperationState.plot_array(
        label="plot_array",
        source="data",
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
    tool.operation_editor.select_section("view")

    aspect_combo = next(
        (
            candidate
            for candidate in tool.findChildren(
                QtWidgets.QComboBox,
                "figureComposerPlotArrayAspectCombo",
            )
            if tool.operation_editor.control_signal_allowed(candidate)
        ),
        None,
    )
    assert aspect_combo is not None
    assert [aspect_combo.itemData(index) for index in range(aspect_combo.count())] == [
        None,
        "auto",
        "equal",
    ]
    assert aspect_combo.currentData() is None

    _activate_combo_index(aspect_combo, aspect_combo.findData("equal"))
    assert tool.tool_status.operations[0].aspect == "equal"
    assert (
        figurecomposer_plot_array._plot_array_kwargs(tool.tool_status.operations[0])[
            "aspect"
        ]
        == "equal"
    )

    _activate_combo_index(aspect_combo, aspect_combo.findData("auto"))
    assert tool.tool_status.operations[0].aspect == "auto"
    assert (
        figurecomposer_plot_array._plot_array_code_kwargs(
            tool.tool_status.operations[0]
        )["aspect"]
        == "auto"
    )

    _activate_combo_index(aspect_combo, aspect_combo.findData(None))
    assert tool.tool_status.operations[0].aspect is None
    assert "aspect" not in figurecomposer_plot_array._plot_array_kwargs(
        tool.tool_status.operations[0]
    )


def test_figure_composer_plot_array_aspect_control_preserves_saved_custom_value(
    qtbot,
) -> None:
    data = _figure_composer_image_source("data").isel(eV=0)
    operation = FigureOperationState.plot_array(
        label="plot_array",
        source="data",
    ).model_copy(update={"aspect": 2.5})
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
    tool.operation_editor.select_section("view")

    aspect_combo = next(
        (
            candidate
            for candidate in tool.findChildren(
                QtWidgets.QComboBox,
                "figureComposerPlotArrayAspectCombo",
            )
            if tool.operation_editor.control_signal_allowed(candidate)
        ),
        None,
    )

    assert aspect_combo is not None
    assert aspect_combo.currentData() is _editor_controls.MIXED_VALUE
    assert tool.tool_status.operations[0].aspect == 2.5

    _activate_combo_index(aspect_combo, aspect_combo.findData("auto"))

    assert tool.tool_status.operations[0].aspect == "auto"


def test_figure_composer_plot_array_aspect_control_shows_mixed_batch_value(
    qtbot,
) -> None:
    data = _figure_composer_image_source("data").isel(eV=0)
    first = FigureOperationState.plot_array(label="first", source="data")
    second = FigureOperationState.plot_array(label="second", source="data").model_copy(
        update={"aspect": "equal"}
    )
    tool = FigureComposerTool.from_sources(
        {"data": data},
        sources=(FigureSourceState(name="data", label="data"),),
        operations=(first, second),
        primary_source="data",
    )
    qtbot.addWidget(tool)

    _select_operation_rows(tool, (0, 1))
    tool.operation_editor.select_section("view")
    combo = tool.operation_editor.stack.currentWidget().findChild(
        QtWidgets.QComboBox, "figureComposerPlotArrayAspectCombo"
    )
    assert combo is not None

    item = typing.cast("typing.Any", combo.model()).item(0)
    assert combo.currentData() is _editor_controls.MIXED_VALUE
    assert item is not None
    assert not item.isEnabled()


def test_figure_composer_plot_array_colorbar_limit_changes_update_recipe(
    qtbot,
) -> None:
    data = _figure_composer_image_source("data").isel(eV=0)
    operation = FigureOperationState.plot_array(
        label="plot_array",
        source="data",
    ).model_copy(update={"colorbar": "right"})
    tool = FigureComposerTool.from_sources(
        {"data": data},
        sources=(FigureSourceState(name="data", label="data"),),
        operations=(operation,),
        primary_source="data",
    )
    qtbot.addWidget(tool)

    figurecomposer_rendering._render_into_figure(tool, tool.figure, sync_visible=False)
    image = tool.figure.axes[0].images[-1]
    assert image._figure_composer_operation_id == operation.operation_id
    assert image._figure_composer_panel_key == (0, 0)
    assert (
        tool._operation_with_colorbar_clim(operation, (1, 0), (-1.0, 1.0)) == operation
    )

    tool._figure_window_colorbar_changed({image: (-1.0, 1.0)})

    updated = tool.tool_status.operations[0]
    assert updated.vmin == -1.0
    assert updated.vmax == 1.0


def test_figure_composer_plot_array_codegen_handles_spaced_dimension(qtbot) -> None:
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
    operation = FigureOperationState.plot_array(
        label="plot_array",
        source="data",
        map_selections=(
            FigureDataSelectionState(source="data", qsel={"Track Shift": 1.0}),
        ),
    )
    tool = FigureComposerTool.from_sources(
        {"data": data},
        sources=(FigureSourceState(name="data", label="data"),),
        operations=(operation,),
        primary_source="data",
    )
    qtbot.addWidget(tool)

    code = tool.generated_code()
    assert 'data_selected = data.qsel(**{"Track Shift": 1.0})' in code
    namespace = _exec_generated_code(code, {"data": data})
    assert "fig" in namespace


def test_figure_composer_plot_array_invalid_target_and_shape(qtbot) -> None:
    image = _figure_composer_image_source("image").isel(eV=0)
    multi_axes = FigureOperationState.plot_array(
        label="plot_array",
        source="image",
        axes=FigureAxesSelectionState(axes=((0, 0), (0, 1))),
    )
    multi_axes_tool = FigureComposerTool.from_sources(
        {"image": image},
        sources=(FigureSourceState(name="image", label="image"),),
        operations=(multi_axes,),
        setup=FigureSubplotsState(nrows=1, ncols=2),
        primary_source="image",
    )
    qtbot.addWidget(multi_axes_tool)
    assert figurecomposer_plot_array._has_invalid_target(
        multi_axes_tool._document, multi_axes
    )
    with pytest.raises(ValueError, match="target axes"):
        multi_axes_tool.generated_code()

    volume = xr.DataArray(
        np.arange(24.0).reshape(2, 3, 4),
        dims=("eV", "beta", "alpha"),
        coords={
            "eV": [0.0, 1.0],
            "beta": [-1.0, 0.0, 1.0],
            "alpha": [-0.5, 0.0, 0.5, 1.0],
        },
        name="volume",
    )
    volume_operation = FigureOperationState.plot_array(
        label="plot_array",
        source="volume",
    )
    volume_tool = FigureComposerTool.from_sources(
        {"volume": volume},
        sources=(FigureSourceState(name="volume", label="volume"),),
        operations=(volume_operation,),
        primary_source="volume",
    )
    qtbot.addWidget(volume_tool)
    assert "3D" in figurecomposer_plot_array._display_text(
        volume_tool, volume_operation
    )
    figurecomposer_rendering._render_into_figure(
        volume_tool, volume_tool.figure, sync_visible=False
    )
    assert (
        "requires a 2D"
        in volume_tool._operation_render_errors[volume_operation.operation_id]
    )


def test_figure_composer_plot_array_helper_edges(qtbot, monkeypatch) -> None:
    image = _figure_composer_image_source("image").isel(eV=0)
    base_tool = FigureComposerTool.from_sources(
        {"image": image},
        sources=(FigureSourceState(name="image", label="image"),),
        operations=(),
        setup=FigureSubplotsState(nrows=1, ncols=2),
        primary_source="image",
    )
    qtbot.addWidget(base_tool)

    operation = FigureOperationState.plot_array(
        label="plot_array",
        source="image",
        map_selections=(
            FigureDataSelectionState(source="image", qsel={"beta": 0.0}),
            FigureDataSelectionState(source="derived", qsel={"beta": 0.0}),
        ),
    )
    assert figurecomposer_operation_metadata.declared_operation_source_names(
        operation
    ) == ("image",)

    empty_operation = FigureOperationState(
        kind=FigureOperationKind.PLOT_ARRAY,
        label="empty",
    )
    empty_tool = typing.cast(
        "FigureComposerTool",
        types.SimpleNamespace(_document=types.SimpleNamespace(source_data={})),
    )
    assert figurecomposer_plot_array._primary_source(empty_operation) is None
    assert (
        figurecomposer_plot_array._plot_array_source_code(empty_tool, empty_operation)
        is None
    )

    missing_selection = FigureOperationState.plot_array(
        label="missing",
        source="missing",
    )
    assert (
        figurecomposer_plot_array._selected_plot_array_data(
            base_tool._document, missing_selection
        )
        is None
    )
    assert (
        figurecomposer_plot_array._plot_array_source_code(base_tool, missing_selection)
        is None
    )
    assert (
        figurecomposer_plot_array._plot_array_code_lines(base_tool, missing_selection)
        == []
    )

    expression_operation = FigureOperationState.plot_array(
        label="expression",
        source="image",
        axes=FigureAxesSelectionState(expression="axs[0, 0]"),
    )
    assert (
        figurecomposer_plot_array._axes_count(base_tool._document, expression_operation)
        == 1
    )

    root = FigureGridSpecGridState(
        nrows=1,
        ncols=2,
        axes=(
            FigureGridSpecAxesState(
                axes_id="left",
                span=FigureGridSpecSpanState(
                    row_start=0, row_stop=1, col_start=0, col_stop=1
                ),
            ),
            FigureGridSpecAxesState(
                axes_id="right",
                span=FigureGridSpecSpanState(
                    row_start=0, row_stop=1, col_start=1, col_stop=2
                ),
            ),
        ),
    )
    grid_tool = FigureComposerTool.from_sources(
        {"image": image},
        sources=(FigureSourceState(name="image", label="image"),),
        operations=(),
        setup=FigureSubplotsState(
            layout_mode="gridspec",
            gridspec=FigureGridSpecLayoutState(root=root),
        ),
        primary_source="image",
    )
    qtbot.addWidget(grid_tool)
    grid_operation = FigureOperationState.plot_array(
        label="grid",
        source="image",
        axes=FigureAxesSelectionState(axes_ids=("left", "missing")),
    )
    assert (
        figurecomposer_plot_array._axes_count(grid_tool._document, grid_operation) == 1
    )

    initial_recipe = base_tool.tool_status
    figurecomposer_plot_array._update_current_source(base_tool.operation_editor, None)
    assert base_tool.tool_status == initial_recipe

    rendered: list[tuple[xr.DataArray, dict[str, typing.Any]]] = []
    monkeypatch.setattr(
        eplt,
        "plot_array",
        lambda arr, **kwargs: rendered.append((arr, kwargs)),
    )
    figurecomposer_plot_array._render_plot_array(
        base_tool,
        FigureOperationState.plot_array(label="missing", source="missing"),
        None,
    )
    assert rendered == []

    _, axs = plt.subplots(1, 2, squeeze=False)
    with pytest.raises(ValueError, match="exactly one target axis"):
        figurecomposer_plot_array._render_plot_array(
            base_tool,
            FigureOperationState.plot_array(
                label="multi_axes",
                source="image",
                axes=FigureAxesSelectionState(axes=((0, 0), (0, 1))),
            ),
            axs,
        )


def test_figure_composer_plot_array_norm_and_callback_helpers(qtbot) -> None:
    power_operation = FigureOperationState.plot_array(
        label="power",
        source="image",
    ).model_copy(
        update={
            "crop": True,
            "colorbar": "right",
            "colorbar_kw": {"pad": 0.01},
            "cmap": "magma",
            "gamma": 0.75,
            "vmin": 0.0,
            "vmax": 1.0,
            "aspect": "equal",
        }
    )

    runtime_kwargs = figurecomposer_plot_array._plot_array_kwargs(power_operation)
    assert runtime_kwargs == {
        "crop": True,
        "colorbar": True,
        "colorbar_kw": {"pad": 0.01},
        "cmap": "magma",
        "gamma": 0.75,
        "vmin": 0.0,
        "vmax": 1.0,
        "aspect": "equal",
    }

    code_kwargs = figurecomposer_plot_array._plot_array_code_kwargs(power_operation)
    assert code_kwargs == runtime_kwargs
    override_operation = power_operation.model_copy(
        update={"extra_kwargs": {"aspect": "auto"}}
    )
    assert (
        figurecomposer_plot_array._plot_array_kwargs(override_operation)["aspect"]
        == "auto"
    )
    assert figurecomposer_plot_array._norm_gamma_value(power_operation) == 0.75

    assert figurecomposer_plot_array._format_aspect_value(None) == ""
    assert figurecomposer_plot_array._format_aspect_value("equal") == "equal"
    assert figurecomposer_plot_array._format_aspect_value(2) == "2"
    assert figurecomposer_plot_array._format_aspect_value((1, 2)) == "(1, 2)"

    norm_operation = power_operation.model_copy(
        update={
            "norm_name": "Normalize",
            "norm_clip": True,
            "gamma": None,
            "norm_gamma": None,
        }
    )
    norm_kwargs = figurecomposer_plot_array._plot_array_kwargs(norm_operation)
    assert isinstance(norm_kwargs["norm"], mcolors.Normalize)
    code_norm_kwargs = figurecomposer_plot_array._plot_array_code_kwargs(norm_operation)
    assert isinstance(code_norm_kwargs["norm"], figurecomposer_text._RawCode)
    assert figurecomposer_plot_array._required_imports(None, norm_operation) == (
        "import erlab.plotting as eplt",
        "import matplotlib.colors as mcolors",
    )

    assert figurecomposer_plot_array._norm_clip_text(True) == "True"
    assert figurecomposer_plot_array._norm_clip_from_text("True") is True
    assert figurecomposer_plot_array._norm_clip_from_text("False") is False
    assert figurecomposer_plot_array._norm_clip_from_text("default") is None

    image = _figure_composer_image_source("image").isel(eV=0)
    editor_tool = FigureComposerTool.from_sources(
        {"image": image},
        sources=(FigureSourceState(name="image", label="image"),),
        operations=(
            FigureOperationState.plot_array(
                label="fake",
                source="image",
            ).model_copy(update={"cmap": "viridis_r"}),
        ),
        primary_source="image",
    )
    qtbot.addWidget(editor_tool)
    editor = editor_tool.operation_editor

    update_vmin = figurecomposer_plot_array._norm_number_update_callback(editor, "vmin")
    update_vmin("  ")
    assert editor_tool.tool_status.operations[0].vmin is None
    update_vmin("1.5")
    assert editor_tool.tool_status.operations[0].vmin == 1.5

    figurecomposer_plot_array._update_current_norm_name(editor, "Normalize")
    operation = editor_tool.tool_status.operations[0]
    assert operation.norm_name == "Normalize"
    assert operation.gamma is None
    assert operation.norm_gamma is None

    figurecomposer_plot_array._update_current_norm_gamma(editor, 0.5)
    operation = editor_tool.tool_status.operations[0]
    assert operation.norm_gamma == 0.5
    assert operation.gamma is None

    figurecomposer_plot_array._update_current_norm_kwargs(
        editor, "gamma=2.0, clip=False, custom='value'"
    )
    operation = editor_tool.tool_status.operations[0]
    assert operation.norm_gamma == 2.0
    assert operation.norm_clip is False
    assert operation.norm_kwargs == {"custom": "value"}

    figurecomposer_plot_array._update_current_cmap(editor, reverse=False)
    assert editor_tool.tool_status.operations[0].cmap == "viridis"
    figurecomposer_plot_array._update_current_cmap(editor, base="plasma")
    assert editor_tool.tool_status.operations[0].cmap == "plasma"

    empty_editor_tool = FigureComposerTool.from_sources(
        {"image": image},
        sources=(FigureSourceState(name="image", label="image"),),
        operations=(),
        primary_source="image",
    )
    qtbot.addWidget(empty_editor_tool)
    figurecomposer_plot_array._update_current_cmap(
        empty_editor_tool.operation_editor,
        base="magma",
    )
    assert empty_editor_tool.tool_status.operations == ()

    assert (
        figurecomposer_plot_array._section_summary(
            editor_tool, "unknown", power_operation
        )
        == ""
    )


def test_figure_composer_toolbar_axes_dialog_updates_plot_array_style(
    qtbot,
) -> None:
    data = _figure_composer_image_source("data").isel(eV=0)
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.plot_array(
                    label="plot_array",
                    source="data",
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
    panel_list = dialog.findChild(
        QtWidgets.QListWidget, "figureComposerPlotSlicesPanelStyleList"
    )
    cmap_combo = dialog.findChild(
        erlab.interactive.colors.ColorMapComboBox, "figureComposerPanelCmapCombo"
    )
    cmap_reverse_check = dialog.findChild(
        QtWidgets.QCheckBox, "figureComposerPanelCmapReverseCheck"
    )
    norm_combo = dialog.findChild(QtWidgets.QComboBox, "figureComposerPanelNormCombo")
    vmin_edit = dialog.findChild(QtWidgets.QLineEdit, "figureComposerPanelVminEdit")
    vmax_edit = dialog.findChild(QtWidgets.QLineEdit, "figureComposerPanelVmaxEdit")
    assert target_combo is not None
    assert target_combo.count() == 1
    assert panel_list is None
    assert cmap_combo is not None
    assert cmap_reverse_check is not None
    assert norm_combo is not None
    assert vmin_edit is not None
    assert vmax_edit is not None

    cmap_combo.load_all()
    assert tool.tool_status.operations[0].cmap is None
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
