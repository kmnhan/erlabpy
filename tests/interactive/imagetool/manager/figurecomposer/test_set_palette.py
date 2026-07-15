# ruff: noqa: F403, F405

from ._common import *


def test_figure_composer_set_palette_editor_preview_and_controls(
    qtbot, monkeypatch
) -> None:
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
    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(0)
    )
    page = tool.operation_editor.stack.currentWidget()
    assert page is not None

    combo = page.findChild(QtWidgets.QComboBox, "figureComposerSetPaletteNameCombo")
    assert combo is not None
    assert combo.isEnabled()
    mode_combo = page.findChild(
        QtWidgets.QComboBox, "figureComposerSetPaletteModeCombo"
    )
    assert mode_combo is not None
    assert mode_combo.currentData() == "named"
    assert combo.findText("jet") == -1
    assert combo.findText("jet_r") == -1
    assert "jet" not in figurecomposer_set_palette._palette_options(
        FigureOperationState.set_palette().model_copy(update={"palette_name": "jet"}),
        sns,
    )
    for palette_name in ("deep6", "tab20b", "Accent", "rocket_r", "husl", "viridis_r"):
        palette_index = combo.findText(palette_name)
        assert palette_index >= 0
        assert not combo.itemIcon(palette_index).isNull()

    opened_urls: list[str] = []

    def record_url(url: QtCore.QUrl) -> bool:
        opened_urls.append(url.toString())
        return True

    monkeypatch.setattr(QtGui.QDesktopServices, "openUrl", record_url)
    docs_button = page.findChild(
        QtWidgets.QToolButton, "figureComposerSetPaletteDocsButton"
    )
    assert docs_button is not None
    assert docs_button.property("figure_palette_doc_url") == (
        "https://seaborn.pydata.org/generated/seaborn.set_palette.html"
    )
    docs_button.click()
    assert opened_urls == [
        "https://seaborn.pydata.org/generated/seaborn.set_palette.html"
    ]

    preview = page.findChild(QtWidgets.QWidget, "figureComposerSetPalettePreview")
    assert preview is not None
    swatches = preview.findChildren(
        QtWidgets.QFrame, "figureComposerSetPalettePreviewSwatch"
    )
    assert swatches
    expected_first = QtGui.QColor.fromRgbF(*sns.color_palette("deep")[0]).name()
    assert swatches[0].property("palette_color") == expected_first
    assert swatches[0].property("palette_tooltip_font_family") == "monospace"
    assert swatches[0].property("palette_tooltip_text_color") in {
        "#000000",
        "#ffffff",
    }
    assert expected_first in swatches[0].toolTip()
    assert "font-family" in swatches[0].toolTip()
    typing.cast("typing.Any", swatches[0]).copy_hex_to_clipboard()
    assert QtWidgets.QApplication.clipboard().text() == expected_first

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


def test_figure_composer_set_palette_custom_colors_editor_and_codegen(
    qtbot,
) -> None:
    sns = pytest.importorskip("seaborn")
    profile = _figure_composer_profile_source("profile")
    palette_operation = FigureOperationState.set_palette().model_copy(
        update={
            "palette_mode": "colors",
            "palette_colors": ("#ff0000", "#00aa00"),
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
    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(0)
    )
    page = tool.operation_editor.stack.currentWidget()
    assert page is not None

    mode_combo = page.findChild(
        QtWidgets.QComboBox, "figureComposerSetPaletteModeCombo"
    )
    assert mode_combo is not None
    assert mode_combo.currentData() == "colors"
    assert (
        page.findChild(QtWidgets.QComboBox, "figureComposerSetPaletteNameCombo") is None
    )
    colors_widget = page.findChild(
        figurecomposer_widgets._ColorListEditorWidget,
        "figureComposerSetPaletteColorsWidget",
    )
    assert colors_widget is not None
    colors_edit = page.findChild(
        QtWidgets.QLineEdit, "figureComposerSetPaletteColorsEdit"
    )
    assert colors_edit is colors_widget.main_edit
    assert colors_widget.colors() == ("#ff0000", "#00aa00")

    preview = page.findChild(QtWidgets.QWidget, "figureComposerSetPalettePreview")
    assert preview is not None
    swatches = preview.findChildren(
        QtWidgets.QFrame, "figureComposerSetPalettePreviewSwatch"
    )
    assert [swatch.property("palette_color") for swatch in swatches[:2]] == [
        "#ff0000",
        "#00aa00",
    ]

    colors_edit.setText("#0000ff, #00ffff")
    colors_edit.setModified(True)
    colors_edit.editingFinished.emit()
    assert tool.tool_status.operations[0].palette_mode == "colors"
    assert tool.tool_status.operations[0].palette_colors == ("#0000ff", "#00ffff")

    figurecomposer_rendering._render_into_figure(tool, tool.figure, sync_visible=False)
    rendered_line = tool.figure.axes[0].lines[0]
    np.testing.assert_allclose(
        mcolors.to_rgb(rendered_line.get_color()),
        mcolors.to_rgb("#0000ff"),
    )

    code = tool.generated_code()
    assert "sns.set_palette(['#0000ff', '#00ffff'])" in code
    namespace: dict[str, typing.Any] = {"profile": profile}
    exec(code, namespace)  # noqa: S102
    generated_line = namespace["fig"].axes[0].lines[0]
    np.testing.assert_allclose(
        mcolors.to_rgb(generated_line.get_color()),
        mcolors.to_rgb("#0000ff"),
    )
    assert sns.color_palette(["#0000ff", "#00ffff"])


def test_figure_composer_set_palette_mode_switch_seeds_custom_colors(qtbot) -> None:
    pytest.importorskip("seaborn")
    profile = _figure_composer_profile_source("profile")
    operation = FigureOperationState.set_palette().model_copy(
        update={"palette_name": "deep", "palette_n_colors": 3}
    )
    recipe = FigureRecipeState(
        sources=(FigureSourceState(name="profile", label="profile"),),
        operations=(operation,),
        primary_source="profile",
    )
    tool = FigureComposerTool(profile, recipe=recipe, source_data={"profile": profile})
    qtbot.addWidget(tool)
    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(0)
    )
    page = tool.operation_editor.stack.currentWidget()
    assert page is not None
    mode_combo = page.findChild(
        QtWidgets.QComboBox, "figureComposerSetPaletteModeCombo"
    )
    assert mode_combo is not None
    colors_index = mode_combo.findData("colors")
    assert colors_index >= 0

    mode_combo.setCurrentIndex(colors_index)
    mode_combo.activated.emit(colors_index)

    operation = tool.tool_status.operations[0]
    assert operation.palette_mode == "colors"
    assert len(operation.palette_colors) == 3
    qtbot.waitUntil(
        lambda: (
            tool.operation_editor.stack.currentWidget() is not None
            and tool.operation_editor.stack.currentWidget().findChild(
                figurecomposer_widgets._ColorListEditorWidget,
                "figureComposerSetPaletteColorsWidget",
            )
            is not None
        ),
        timeout=1000,
    )
    page = tool.operation_editor.stack.currentWidget()
    assert page is not None
    colors_widget = page.findChild(
        figurecomposer_widgets._ColorListEditorWidget,
        "figureComposerSetPaletteColorsWidget",
    )
    assert colors_widget is not None
    assert colors_widget.colors() == operation.palette_colors


def test_figure_composer_set_palette_swatch_tooltip_contrast(qtbot) -> None:
    dark = figurecomposer_set_palette._PaletteSwatch(QtGui.QColor("#000000"), 0)
    light = figurecomposer_set_palette._PaletteSwatch(QtGui.QColor("#ffffff"), 1)
    qtbot.addWidget(dark)
    qtbot.addWidget(light)

    assert dark.property("palette_tooltip_text_color") == "#ffffff"
    assert light.property("palette_tooltip_text_color") == "#000000"
    assert dark.property("palette_tooltip_font_family") == "monospace"
    assert light.property("palette_tooltip_font_family") == "monospace"
    assert "background-color: #000000" in dark.toolTip()
    assert "color: #ffffff" in dark.toolTip()
    assert "background-color: #ffffff" in light.toolTip()
    assert "color: #000000" in light.toolTip()
    assert "monospace" in dark.toolTip()
    assert "monospace" in light.toolTip()


def test_figure_composer_set_palette_helper_fallbacks(qtbot, monkeypatch) -> None:
    sns = pytest.importorskip("seaborn")
    swatch = figurecomposer_set_palette._PaletteSwatch(QtGui.QColor("#336699"), 0)
    qtbot.addWidget(swatch)
    swatch.contextMenuEvent(None)

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

    monkeypatch.setattr(builtins, "__import__", import_without_seaborn)
    assert figurecomposer_set_palette._import_seaborn() is None

    custom_operation = FigureOperationState.set_palette().model_copy(
        update={"palette_name": "custom_lab_palette"}
    )
    options = figurecomposer_set_palette._palette_options(custom_operation, None)
    assert "custom_lab_palette" in options
    assert "jet" not in options

    assert (
        figurecomposer_set_palette._palette_colors(
            sns, "not-a-real-palette", None, None
        )
        == ()
    )
    assert figurecomposer_set_palette._palette_icon(None, "deep").isNull()

    class EmptyPalette:
        @staticmethod
        def color_palette() -> tuple[typing.Any, ...]:
            return ()

    fig = plt.figure()
    try:
        fig.add_subplot()
        figurecomposer_set_palette._apply_palette_to_existing_axes(fig, EmptyPalette)
    finally:
        plt.close(fig)

    colors_operation = FigureOperationState.set_palette().model_copy(
        update={"palette_mode": "colors", "palette_colors": ()}
    )
    assert (
        figurecomposer_set_palette._palette_display_text(colors_operation)
        == "Custom colors"
    )
    assert figurecomposer_set_palette._palette_call_code(colors_operation) is None
    assert (
        figurecomposer_set_palette._code_lines(
            typing.cast("typing.Any", None), colors_operation
        )
        == []
    )

    monkeypatch.setattr(figurecomposer_set_palette, "_import_seaborn", lambda: None)
    operation = figurecomposer_set_palette._create_operation(
        typing.cast("typing.Any", None)
    )
    assert operation.label == "set palette"
    assert figurecomposer_set_palette._display_text(
        typing.cast("typing.Any", None), operation
    ).startswith("Skipped Set Palette:")
    assert "Install seaborn" in figurecomposer_set_palette._tooltip(
        typing.cast("typing.Any", None), operation
    )
    assert (
        figurecomposer_set_palette._section_summary(
            typing.cast("typing.Any", None), "other", operation
        )
        == ""
    )


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
    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(0)
    )
    page = tool.operation_editor.stack.currentWidget()
    assert page is not None

    combo = page.findChild(QtWidgets.QComboBox, "figureComposerSetPaletteNameCombo")
    count_spin = page.findChild(QtWidgets.QSpinBox, "figureComposerSetPaletteCountSpin")
    desat_spin = page.findChild(
        QtWidgets.QDoubleSpinBox, "figureComposerSetPaletteSaturationSpin"
    )
    color_codes = page.findChild(
        QtWidgets.QCheckBox, "figureComposerSetPaletteColorCodesCheck"
    )
    docs_button = page.findChild(
        QtWidgets.QToolButton, "figureComposerSetPaletteDocsButton"
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
    assert docs_button is not None
    assert docs_button.isEnabled()
    assert preview is not None
    assert not preview.isEnabled()
    assert message is not None
    assert message.property("missing_dependency") == "seaborn"
    item = tool.operation_panel.operation_list.topLevelItem(0)
    assert item is not None

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
