"""Plot Slices integration behavior tests."""

from ._plot_slices_common import (
    Figure,
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
    _activate_combo_index,
    _activate_combo_text,
    _drag_widget,
    _figure_composer_image_source,
    _operation_section_button,
    _operation_section_buttons,
    _plot_source_checks,
    _select_operation_rows,
    _set_figure_stylesheets,
    erlab,
    figurecomposer_norms,
    figurecomposer_rendering,
    figurecomposer_toolbar_dialogs,
    mpl,
    mpl_style,
    np,
    plot_slices_panel_style_editor,
    typing,
    warnings,
    xr,
)


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
