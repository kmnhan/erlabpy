"""Plot Slices styles behavior tests."""

from ._plot_slices_common import (
    FigureAxesSelectionState,
    FigureComposerTool,
    FigureOperationState,
    FigurePlotSlicesPanelStyleState,
    FigureRecipeState,
    FigureSourceState,
    FigureSubplotsState,
    Path,
    QtCore,
    QtWidgets,
    _activate_combo_text,
    _figure_composer_image_source,
    _select_operation_rows,
    _set_figure_stylesheets,
    erlab,
    figurecomposer_norms,
    figurecomposer_rendering,
    figurecomposer_tool_module,
    figurecomposer_toolbar_dialogs,
    mpl,
    mpl_style,
    np,
    plot_slices_editor,
    plot_slices_model,
    plot_slices_panel_style_editor,
    pytest,
    typing,
    warnings,
    xr,
)


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
