# ruff: noqa: F403, F405

from ._common import *


def test_figure_composer_plot_array_source_selector_updates_selection(qtbot) -> None:
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
    tool.operation_list.setCurrentRow(0)
    tool._select_step_section("sources")

    combo = next(
        (
            candidate
            for candidate in tool.findChildren(
                QtWidgets.QComboBox, "figureComposerPlotArraySourceCombo"
            )
            if candidate.property("figure_composer_editor_generation")
            == tool._operation_editor_generation
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
    assert updated.map_selections == (
        FigureDataSelectionState(source="second_source", qsel={"eV": 0.0}),
    )


def test_figure_composer_plot_array_selection_editor_updates_selection(qtbot) -> None:
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
    operation = FigureOperationState.plot_array(
        label="plot_array",
        source="data",
        map_selections=(
            FigureDataSelectionState(
                source="data",
                qsel={"eV": 0.0},
                mean_dims=("hv",),
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
    tool.operation_list.setCurrentRow(0)
    tool._select_step_section("selection")

    def current_dimension_widget(
        widget_type: type[QtWidgets.QWidget], dim: str
    ) -> QtWidgets.QWidget:
        widget = next(
            (
                candidate
                for candidate in tool.findChildren(widget_type)
                if candidate.property("figure_composer_plot_array_dim") == dim
                if candidate.property("figure_composer_editor_generation")
                == tool._operation_editor_generation
            ),
            None,
        )
        assert widget is not None
        return widget

    def activate_dimension_mode(dim: str, mode: str) -> None:
        combo = typing.cast(
            "QtWidgets.QComboBox",
            current_dimension_widget(QtWidgets.QComboBox, dim),
        )
        index = combo.findData(mode)
        assert index >= 0
        combo.setCurrentIndex(index)
        combo.activated.emit(index)

    def current_dimension_value(dim: str) -> QtWidgets.QLineEdit:
        return typing.cast(
            "QtWidgets.QLineEdit",
            current_dimension_widget(QtWidgets.QLineEdit, dim),
        )

    summary = tool.findChild(
        QtWidgets.QLabel, "figureComposerPlotArraySelectionSummary"
    )
    assert summary is not None
    assert "Input dims: hv, eV, beta, alpha" in summary.text()
    assert "Plotted dims: beta, alpha" in summary.text()

    eV_mode = typing.cast(
        "QtWidgets.QComboBox",
        current_dimension_widget(QtWidgets.QComboBox, "eV"),
    )
    assert eV_mode.currentData() == "qsel"
    qsel_edit = current_dimension_value("eV")
    qsel_edit.setText("1.0")
    qsel_edit.setModified(True)
    qsel_edit.editingFinished.emit()

    updated = tool.tool_status.operations[0]
    assert updated.map_selections == (
        FigureDataSelectionState(
            source="data",
            qsel={"eV": 1.0},
            mean_dims=("hv",),
        ),
    )

    activate_dimension_mode("eV", "isel")
    updated = tool.tool_status.operations[0]
    assert updated.map_selections == (
        FigureDataSelectionState(
            source="data",
            isel={"eV": 1.0},
            mean_dims=("hv",),
        ),
    )
    activate_dimension_mode("eV", "qsel")
    updated = tool.tool_status.operations[0]
    assert updated.map_selections == (
        FigureDataSelectionState(
            source="data",
            qsel={"eV": 1.0},
            mean_dims=("hv",),
        ),
    )

    activate_dimension_mode("hv", "keep")
    activate_dimension_mode("hv", "isel")
    isel_edit = current_dimension_value("hv")
    isel_edit.setText("1")
    isel_edit.setModified(True)
    isel_edit.editingFinished.emit()

    updated = tool.tool_status.operations[0]
    assert updated.map_selections == (
        FigureDataSelectionState(
            source="data",
            isel={"hv": 1},
            qsel={"eV": 1.0},
        ),
    )


def test_figure_composer_plot_array_qsel_to_keep_clears_selection(qtbot) -> None:
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
    tool.operation_list.setCurrentRow(0)
    tool._select_step_section("selection")

    def current_dimension_widget(
        widget_type: type[QtWidgets.QWidget], dim: str
    ) -> QtWidgets.QWidget:
        widget = next(
            (
                candidate
                for candidate in tool.findChildren(widget_type)
                if candidate.property("figure_composer_plot_array_dim") == dim
                if candidate.property("figure_composer_editor_generation")
                == tool._operation_editor_generation
            ),
            None,
        )
        assert widget is not None
        return widget

    def activate_dimension_mode(dim: str, mode: str) -> None:
        combo = typing.cast(
            "QtWidgets.QComboBox",
            current_dimension_widget(QtWidgets.QComboBox, dim),
        )
        index = combo.findData(mode)
        assert index >= 0
        combo.setCurrentIndex(index)
        combo.activated.emit(index)

    activate_dimension_mode("eV", "qsel")
    qsel_edit = typing.cast(
        "QtWidgets.QLineEdit",
        current_dimension_widget(QtWidgets.QLineEdit, "eV"),
    )
    assert qsel_edit.isEnabled()
    assert not _plot_array_dimension_edit(tool, "eV", "value").isHidden()
    assert not _plot_array_dimension_edit(tool, "eV", "width").isHidden()
    qsel_edit.setText("0.0")
    qsel_edit.setModified(True)
    qsel_edit.editingFinished.emit()

    updated = tool.tool_status.operations[0]
    assert updated.map_selections == (
        FigureDataSelectionState(source="data", qsel={"eV": 0.0}),
    )

    activate_dimension_mode("eV", "keep")

    updated = tool.tool_status.operations[0]
    assert updated.sources == ("data",)
    assert updated.map_selections == ()
    value_edit = _plot_array_dimension_edit(tool, "eV", "value")
    width_edit = _plot_array_dimension_edit(tool, "eV", "width")
    assert value_edit.isHidden()
    assert width_edit.isHidden()
    assert value_edit.text() == ""


def test_figure_composer_plot_array_keep_clears_stale_qsel_input_error(
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
    tool.operation_list.setCurrentRow(0)
    tool._select_step_section("selection")

    _activate_plot_array_dimension_mode(tool, "eV", "qsel")
    qsel_edit = _plot_array_dimension_edit(tool, "eV", "value")
    qsel_edit.setText("slice(-0.5, 0.5)")
    qsel_edit.setModified(True)
    qsel_edit.editingFinished.emit()

    updated = tool.tool_status.operations[0]
    assert updated.map_selections == (
        FigureDataSelectionState(source="data", qsel={"eV": slice(-0.5, 0.5)}),
    )

    qsel_edit = _plot_array_dimension_edit(tool, "eV", "value")
    qsel_edit.setText("")
    qsel_edit.setModified(True)
    qsel_edit.editingFinished.emit()
    assert tool._operation_input_errors

    _activate_plot_array_dimension_mode(tool, "eV", "keep")

    updated = tool.tool_status.operations[0]
    assert updated.sources == ("data",)
    assert updated.map_selections == ()
    assert tool._operation_input_errors == {}


def test_figure_composer_plot_array_qsel_width_editor_updates_selection(
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

    selected = figurecomposer_plot_array._safe_selected_plot_array_data(tool, operation)
    assert selected is not None
    xr.testing.assert_identical(selected, data.qsel(eV=0.0, eV_width=0.2))
    assert "data.qsel(eV=0.0, eV_width=0.2)" in tool.generated_code()

    tool.operation_list.setCurrentRow(0)
    tool._select_step_section("selection")
    mode_combo = typing.cast(
        "QtWidgets.QComboBox",
        _plot_array_dimension_widget(tool, QtWidgets.QComboBox, "eV"),
    )
    value_edit = _plot_array_dimension_edit(tool, "eV", "value")
    width_edit = _plot_array_dimension_edit(tool, "eV", "width")
    assert not value_edit.isHidden()
    assert not width_edit.isHidden()
    assert value_edit.text() == "0.0"
    assert width_edit.text() == "0.2"
    assert mode_combo.toolTip()
    assert value_edit.toolTip()
    assert width_edit.toolTip()
    assert len({mode_combo.toolTip(), value_edit.toolTip(), width_edit.toolTip()}) == 3

    width_edit.setText("0.5")
    width_edit.setModified(True)
    width_edit.editingFinished.emit()
    updated = tool.tool_status.operations[0]
    assert updated.map_selections == (
        FigureDataSelectionState(
            source="data",
            qsel={"eV": 0.0, "eV_width": 0.5},
        ),
    )

    _activate_plot_array_dimension_mode(tool, "eV", "isel")
    updated = tool.tool_status.operations[0]
    assert updated.map_selections == (
        FigureDataSelectionState(source="data", isel={"eV": 0.0}),
    )
    assert not _plot_array_dimension_edit(tool, "eV", "value").isHidden()
    assert _plot_array_dimension_edit(tool, "eV", "width").isHidden()

    _activate_plot_array_dimension_mode(tool, "eV", "qsel")
    width_edit = _plot_array_dimension_edit(tool, "eV", "width")
    width_edit.setText("0.1")
    width_edit.setModified(True)
    width_edit.editingFinished.emit()
    updated = tool.tool_status.operations[0]
    assert updated.map_selections == (
        FigureDataSelectionState(
            source="data",
            qsel={"eV": 0.0, "eV_width": 0.1},
        ),
    )

    _activate_plot_array_dimension_mode(tool, "eV", "mean")
    updated = tool.tool_status.operations[0]
    assert updated.map_selections == (
        FigureDataSelectionState(source="data", mean_dims=("eV",)),
    )
    assert _plot_array_dimension_edit(tool, "eV", "value").isHidden()
    assert _plot_array_dimension_edit(tool, "eV", "width").isHidden()


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

    assert "invalid selection" in figurecomposer_plot_array._display_text(
        tool, operation
    )

    tool.operation_list.setCurrentRow(0)
    tool._select_step_section("selection")

    summary = tool.findChild(
        QtWidgets.QLabel, "figureComposerPlotArraySelectionSummary"
    )
    assert summary is not None
    assert "missing" in summary.text()


def test_figure_composer_plot_array_selection_helper_edges(qtbot) -> None:
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
            FigureOperationState.plot_array(label="plot_array", source="data"),
        ),
        primary_source="data",
    )
    qtbot.addWidget(tool)
    figurecomposer_plot_array._update_current_selection_source(tool, "data")
    assert tool.tool_status.operations[0].sources == ("data",)
    assert tool.tool_status.operations[0].map_selections == ()

    missing_operation = FigureOperationState.plot_array(
        label="missing", source="missing"
    )
    assert (
        figurecomposer_plot_array._plot_array_source_data(tool, missing_operation)
        is None
    )
    assert "missing" in figurecomposer_plot_array._display_text(tool, missing_operation)

    no_update_tool = types.SimpleNamespace(
        _update_operations=lambda *_args, **_kwargs: pytest.fail(
            "None source should not update"
        )
    )
    figurecomposer_plot_array._update_current_selection_source(
        typing.cast("FigureComposerTool", no_update_tool),
        None,
    )
    figurecomposer_plot_array._update_current_selection_dimension(
        typing.cast("FigureComposerTool", no_update_tool),
        "x",
        "bad",
    )

    with pytest.raises(figurecomposer_text.FigureComposerInputError):
        figurecomposer_plot_array._plot_array_selection_value_from_text("")
    with pytest.raises(figurecomposer_text.FigureComposerInputError):
        figurecomposer_plot_array._plot_array_selection_value_from_text("slice(")
    with pytest.raises(figurecomposer_text.FigureComposerInputError):
        figurecomposer_plot_array._plot_array_selection_width_from_text("slice(0, 1)")

    selection = FigureDataSelectionState(
        source="data",
        isel={"x": 0},
        qsel={"y": 0.0},
    )
    assert figurecomposer_plot_array._plot_array_selection_with_dimension(
        selection,
        "x",
        "mean",
    ) == FigureDataSelectionState(
        source="data",
        qsel={"y": 0.0},
        mean_dims=("x",),
    )
    assert figurecomposer_plot_array._plot_array_selection_with_dimension(
        FigureDataSelectionState(source="data"),
        "x",
        "isel",
        slice(0, 2),
    ) == FigureDataSelectionState(source="data", isel={"x": slice(0, 2)})

    empty_selection = FigureDataSelectionState(
        source="data",
        qsel={"x": 0.0, "x_width": 0.1},
    )
    assert figurecomposer_plot_array._plot_array_selection_with_dimension(
        empty_selection,
        "x",
        "keep",
    ) == FigureDataSelectionState(source="data")
    width_only = FigureDataSelectionState(source="data", qsel={"x_width": 0.1})
    assert (
        figurecomposer_plot_array._plot_array_selection_dim_mode(width_only, "x")
        == "qsel"
    )
    assert (
        figurecomposer_plot_array._plot_array_selection_dim_width_text(width_only, "x")
        == "0.1"
    )

    combo = figurecomposer_plot_array._plot_array_selection_mode_combo(
        tool,
        current=None,
        mixed=True,
        parent=tool,
    )
    assert combo.currentData() is _editor_controls.MIXED_VALUE


def test_figure_composer_plot_array_selection_page_empty_sources(qtbot) -> None:
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
    tool.operation_list.setCurrentRow(0)
    tool._select_step_section("selection")
    assert (
        tool.findChild(
            QtWidgets.QLabel,
            "figureComposerPlotArraySelectionDimensionsMessage",
        )
        is not None
    )

    scalar = xr.DataArray(1.0, name="scalar")
    scalar_tool = FigureComposerTool.from_sources(
        {"scalar": scalar},
        sources=(FigureSourceState(name="scalar", label="scalar"),),
        operations=(FigureOperationState.plot_array(label="scalar", source="scalar"),),
        primary_source="scalar",
    )
    qtbot.addWidget(scalar_tool)
    scalar_tool.operation_list.setCurrentRow(0)
    scalar_tool._select_step_section("selection")
    assert (
        scalar_tool.findChild(
            QtWidgets.QLabel,
            "figureComposerPlotArraySelectionDimensionsMessage",
        )
        is not None
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
    mixed_tool._select_step_section("selection")
    assert (
        mixed_tool.findChild(
            QtWidgets.QLabel,
            "figureComposerPlotArraySelectionDimensionsMessage",
        )
        is not None
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
    tool.operation_list.setCurrentRow(0)
    tool._select_step_section("colors")

    cmap_combo = next(
        (
            candidate
            for candidate in tool.findChildren(
                erlab.interactive.colors.ColorMapComboBox,
                "figureComposerPlotArrayCmapCombo",
            )
            if candidate.property("figure_composer_editor_generation")
            == tool._operation_editor_generation
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
    tool.operation_list.setCurrentRow(0)
    tool._select_step_section("colors")

    cmap_combo = next(
        (
            candidate
            for candidate in tool.findChildren(
                erlab.interactive.colors.ColorMapComboBox,
                "figureComposerPlotArrayCmapCombo",
            )
            if candidate.property("figure_composer_editor_generation")
            == tool._operation_editor_generation
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
        tool.operation_list.setCurrentRow(0)
        tool._select_step_section("colors")

        cmap_combo = next(
            (
                candidate
                for candidate in tool.findChildren(
                    erlab.interactive.colors.ColorMapComboBox,
                    "figureComposerPlotArrayCmapCombo",
                )
                if candidate.property("figure_composer_editor_generation")
                == tool._operation_editor_generation
            ),
            None,
        )
        cmap_reverse_check = next(
            (
                candidate
                for candidate in tool.findChildren(
                    QtWidgets.QCheckBox, "figureComposerPlotArrayCmapReverseCheck"
                )
                if candidate.property("figure_composer_editor_generation")
                == tool._operation_editor_generation
            ),
            None,
        )
        assert cmap_combo is not None
        assert cmap_reverse_check is not None
        assert cmap_combo.currentText() == "plasma"
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
    tool.operation_list.setCurrentRow(0)
    tool._select_step_section("colors")
    cmap_combo = next(
        (
            candidate
            for candidate in tool.findChildren(
                erlab.interactive.colors.ColorMapComboBox,
                "figureComposerPlotArrayCmapCombo",
            )
            if candidate.property("figure_composer_editor_generation")
            == tool._operation_editor_generation
        ),
        None,
    )
    assert cmap_combo is not None
    assert cmap_combo.currentText() == "magma"

    missing_cmap_tool = FigureComposerTool.from_sources(
        {"data": data},
        sources=(FigureSourceState(name="data", label="data"),),
        operations=(explicit.model_copy(update={"cmap": "missing_colormap_name"}),),
        primary_source="data",
    )
    qtbot.addWidget(missing_cmap_tool)
    missing_cmap_tool.operation_list.setCurrentRow(0)
    missing_cmap_tool._select_step_section("colors")
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
            explicit.model_copy(update={"cmap": "viridis"}),
        ),
        primary_source="data",
    )
    qtbot.addWidget(mixed_tool)
    _select_operation_rows(mixed_tool, (0, 1))
    mixed_tool._select_step_section("colors")
    mixed_cmap_combo = next(
        (
            candidate
            for candidate in mixed_tool.findChildren(
                erlab.interactive.colors.ColorMapComboBox,
                "figureComposerPlotArrayCmapCombo",
            )
            if candidate.property("figure_composer_editor_generation")
            == mixed_tool._operation_editor_generation
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
        for action in tool.add_step_menu.actions()
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
    assert not figurecomposer_plot_array._has_invalid_target(tool, operation)

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

    def capture_plot_array(arr, **kwargs):
        captured.append((arr, kwargs))

    monkeypatch.setattr(eplt, "plot_array", capture_plot_array)
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
    assert "eplt.plot_array(data.qsel(eV=0.0).T" in code
    namespace = _exec_generated_code(code, {"data": data})
    assert "fig" in namespace


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
    tool.operation_list.setCurrentRow(0)
    tool._select_step_section("view")

    aspect_combo = next(
        (
            candidate
            for candidate in tool.findChildren(
                QtWidgets.QComboBox,
                "figureComposerPlotArrayAspectCombo",
            )
            if candidate.property("figure_composer_editor_generation")
            == tool._operation_editor_generation
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
    tool.operation_list.setCurrentRow(0)
    tool._select_step_section("view")

    aspect_combo = next(
        (
            candidate
            for candidate in tool.findChildren(
                QtWidgets.QComboBox,
                "figureComposerPlotArrayAspectCombo",
            )
            if candidate.property("figure_composer_editor_generation")
            == tool._operation_editor_generation
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
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)
    operation = FigureOperationState.plot_array(
        label="plot_array",
        source="data",
    )

    class _BatchTool:
        def _batch_is_mixed(
            self,
            _operation: FigureOperationState,
            _value: Callable[[FigureOperationState], object],
        ) -> bool:
            return True

        def _mark_editor_control(self, _widget: QtWidgets.QWidget) -> None:
            return

        def _connect_editor_signal(
            self,
            _owner: QtWidgets.QWidget,
            signal: object,
            slot: Callable[..., None],
        ) -> None:
            signal.connect(slot)

        def _update_current_operation(self, **_updates: object) -> None:
            return

    combo = figurecomposer_plot_array._plot_array_aspect_combo(
        typing.cast("FigureComposerTool", _BatchTool()),
        operation,
        parent=parent,
    )

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
    assert 'data.qsel(**{"Track Shift": 1.0})' in code
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
    assert figurecomposer_plot_array._has_invalid_target(multi_axes_tool, multi_axes)
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
    assert figurecomposer_plot_array._source_names(operation) == ("image", "derived")

    empty_operation = FigureOperationState(
        kind=FigureOperationKind.PLOT_ARRAY,
        label="empty",
    )
    empty_tool = typing.cast(
        "FigureComposerTool", types.SimpleNamespace(_source_data={})
    )
    assert figurecomposer_plot_array._primary_source(empty_operation) is None
    assert (
        figurecomposer_plot_array._plot_array_source_code(empty_tool, empty_operation)
        is None
    )

    missing_selection = FigureOperationState.plot_array(
        label="missing",
        source="image",
        map_selections=(FigureDataSelectionState(source="missing", qsel={"eV": 0.0}),),
    )
    assert (
        figurecomposer_plot_array._selected_plot_array_data(
            base_tool, missing_selection
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
    assert figurecomposer_plot_array._axes_count(base_tool, expression_operation) == 1

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
    assert figurecomposer_plot_array._axes_count(grid_tool, grid_operation) == 1

    no_update_tool = types.SimpleNamespace(
        _update_operations=lambda *_args, **_kwargs: pytest.fail(
            "None source should not update"
        )
    )
    figurecomposer_plot_array._update_current_source(
        typing.cast("FigureComposerTool", no_update_tool), None
    )

    rendered: list[tuple[xr.DataArray, dict[str, typing.Any]]] = []
    monkeypatch.setattr(
        eplt,
        "plot_array",
        lambda arr, **kwargs: rendered.append((arr, kwargs)),
    )
    figurecomposer_plot_array._render_plot_array(
        empty_tool,
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


def test_figure_composer_plot_array_norm_and_callback_helpers() -> None:
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

    class _FakeTool:
        def __init__(self) -> None:
            self.operation = FigureOperationState.plot_array(
                label="fake",
                source="image",
            ).model_copy(update={"cmap": "viridis_r"})
            self.updates: list[dict[str, typing.Any]] = []
            self.rebuild_updates: list[dict[str, typing.Any]] = []

        def _update_current_operation(self, **updates: typing.Any) -> None:
            self.updates.append(updates)
            self.operation = self.operation.model_copy(update=updates)

        def _update_current_operation_rebuild(self, **updates: typing.Any) -> None:
            self.rebuild_updates.append(updates)
            self.operation = self.operation.model_copy(update=updates)

        def _current_operation(self) -> tuple[int, FigureOperationState] | None:
            return 0, self.operation

    fake_tool = _FakeTool()
    cast_tool = typing.cast("FigureComposerTool", fake_tool)
    update_vmin = figurecomposer_plot_array._norm_number_update_callback(
        cast_tool, "vmin"
    )
    update_vmin("  ")
    update_vmin("1.5")
    figurecomposer_plot_array._update_current_norm_name(cast_tool, "Normalize")
    figurecomposer_plot_array._update_current_norm_gamma(cast_tool, 0.5)
    figurecomposer_plot_array._update_current_norm_kwargs(
        cast_tool, "gamma=2.0, clip=False, custom='value'"
    )
    figurecomposer_plot_array._update_current_cmap(cast_tool, reverse=False)
    figurecomposer_plot_array._update_current_cmap(cast_tool, base="plasma")
    assert fake_tool.updates[:2] == [{"vmin": None}, {"vmin": 1.5}]
    assert {"norm_gamma": 0.5, "gamma": None} in fake_tool.updates
    assert fake_tool.updates[-2:] == [{"cmap": "viridis"}, {"cmap": "plasma"}]
    assert fake_tool.rebuild_updates[0] == {
        "norm_name": "Normalize",
        "gamma": None,
        "norm_gamma": None,
    }
    assert fake_tool.rebuild_updates[-1] == {
        "norm_gamma": 2.0,
        "norm_clip": False,
        "norm_kwargs": {"custom": "value"},
    }

    none_tool = types.SimpleNamespace(_current_operation=lambda: None)
    figurecomposer_plot_array._update_current_cmap(
        typing.cast("FigureComposerTool", none_tool),
        base="magma",
    )

    assert (
        figurecomposer_plot_array._section_summary(
            typing.cast("FigureComposerTool", none_tool), "unknown", power_operation
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
