# ruff: noqa: F403, F405

import erlab.interactive._figurecomposer._ui._color_widgets as color_widgets

from ._common import *


def _set_palette_tool(qtbot, operation: FigureOperationState) -> FigureComposerTool:
    profile = _figure_composer_profile_source("profile")
    recipe = FigureRecipeState(
        sources=(FigureSourceState(name="profile", label="profile"),),
        operations=(operation,),
        primary_source="profile",
    )
    tool = FigureComposerTool(
        profile,
        recipe=recipe,
        source_data={"profile": profile},
    )
    qtbot.addWidget(tool)
    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(0)
    )
    return tool


def _set_palette_page(tool: FigureComposerTool) -> QtWidgets.QWidget:
    page = tool.operation_editor.stack.currentWidget()
    assert page is not None
    return page


def _activate_combo_data(combo: QtWidgets.QComboBox, data: typing.Any) -> None:
    index = combo.findData(data)
    assert index >= 0
    combo.setCurrentIndex(index)
    combo.activated.emit(index)


def test_figure_composer_generated_palette_state_validation_and_roundtrip() -> None:
    cubehelix = FigureCubehelixPaletteState(
        n_colors=12,
        start=1.2,
        rot=-0.6,
        gamma=1.4,
        hue=0.7,
        light=0.9,
        dark=0.1,
        reverse=True,
    )
    light = FigureSequentialPaletteState(input="rgb", color=(0.2, 0.4, 0.8), n_colors=7)
    operation = FigureOperationState.set_palette().model_copy(
        update={
            "palette_mode": "light",
            "palette_cubehelix": cubehelix,
            "palette_light": light,
        }
    )

    restored = FigureOperationState.model_validate_json(operation.model_dump_json())
    assert restored == operation
    assert restored.palette_cubehelix == cubehelix
    assert restored.palette_light == light

    legacy = FigureOperationState.model_validate(
        {"kind": "set_palette", "label": "palette"}
    )
    assert legacy.palette_mode == "named"
    assert legacy.palette_cubehelix == FigureCubehelixPaletteState()
    assert legacy.palette_light == FigureSequentialPaletteState()
    assert legacy.palette_dark == FigureSequentialPaletteState()

    with pytest.raises(ValueError, match="less than or equal to 16"):
        FigureCubehelixPaletteState(n_colors=17)
    with pytest.raises(ValueError, match="outside the husl range"):
        FigureSequentialPaletteState(input="husl", color=(360.0, 50.0, 50.0))
    with pytest.raises(ValueError, match="outside the rgb range"):
        FigureSequentialPaletteState(input="rgb", color=(1.1, 0.5, 0.5))


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
        color_widgets._ColorListEditorWidget,
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
                color_widgets._ColorListEditorWidget,
                "figureComposerSetPaletteColorsWidget",
            )
            is not None
        ),
        timeout=1000,
    )
    page = tool.operation_editor.stack.currentWidget()
    assert page is not None
    colors_widget = page.findChild(
        color_widgets._ColorListEditorWidget,
        "figureComposerSetPaletteColorsWidget",
    )
    assert colors_widget is not None
    assert colors_widget.colors() == operation.palette_colors


def test_figure_composer_set_palette_cubehelix_editor_and_custom_seed(qtbot) -> None:
    sns = pytest.importorskip("seaborn")
    operation = FigureOperationState.set_palette().model_copy(
        update={
            "palette_mode": "cubehelix",
            "palette_cubehelix": FigureCubehelixPaletteState(n_colors=5),
        }
    )
    tool = _set_palette_tool(qtbot, operation)
    page = _set_palette_page(tool)

    mode_combo = page.findChild(
        QtWidgets.QComboBox, "figureComposerSetPaletteModeCombo"
    )
    assert mode_combo is not None
    assert [mode_combo.itemData(index) for index in range(mode_combo.count())] == [
        "named",
        "colors",
        "cubehelix",
        "light",
        "dark",
    ]
    assert (
        page.findChild(QtWidgets.QComboBox, "figureComposerSetPaletteNameCombo") is None
    )
    assert (
        page.findChild(QtWidgets.QSpinBox, "figureComposerSetPaletteCountSpin") is None
    )
    assert (
        page.findChild(QtWidgets.QCheckBox, "figureComposerSetPaletteColorCodesCheck")
        is None
    )

    count = page.findChild(
        figurecomposer_set_palette._PaletteSliderWidget,
        "figureComposerSetPaletteCubehelixNColorsControl",
    )
    rotation = page.findChild(
        figurecomposer_set_palette._PaletteSliderWidget,
        "figureComposerSetPaletteCubehelixRotationControl",
    )
    reverse = page.findChild(
        QtWidgets.QCheckBox, "figureComposerSetPaletteCubehelixReverseCheck"
    )
    docs_button = page.findChild(
        QtWidgets.QToolButton, "figureComposerSetPaletteDocsButton"
    )
    assert count is not None
    assert count.slider.minimum() == 2
    assert count.slider.maximum() == 16
    assert count.value() == 5
    assert rotation is not None
    assert rotation.spin.minimum() == pytest.approx(-1.0)
    assert rotation.spin.maximum() == pytest.approx(1.0)
    assert reverse is not None
    assert docs_button is not None
    assert docs_button.property("figure_palette_doc_url") == (
        "https://seaborn.pydata.org/generated/seaborn.cubehelix_palette.html"
    )

    count.slider.setValue(7)
    rotation.slider.setValue(-25)
    reverse.click()
    assert tool.tool_status.operations[0].palette_cubehelix.n_colors == 7
    assert tool.tool_status.operations[0].palette_cubehelix.rot == pytest.approx(-0.25)
    assert tool.tool_status.operations[0].palette_cubehelix.reverse is True

    preview = page.findChild(QtWidgets.QWidget, "figureComposerSetPalettePreview")
    assert preview is not None
    swatches = preview.findChildren(
        QtWidgets.QFrame, "figureComposerSetPalettePreviewSwatch"
    )
    expected = sns.cubehelix_palette(
        n_colors=7,
        start=0.0,
        rot=-0.25,
        gamma=1.0,
        hue=0.8,
        light=0.85,
        dark=0.15,
        reverse=True,
    )
    assert [swatch.property("palette_color") for swatch in swatches] == [
        QtGui.QColor.fromRgbF(*color).name() for color in expected
    ]

    _activate_combo_data(mode_combo, "colors")
    expected_hex = tuple(QtGui.QColor.fromRgbF(*color).name() for color in expected)
    current = tool.tool_status.operations[0]
    assert current.palette_mode == "colors"
    assert current.palette_colors == expected_hex
    qtbot.waitUntil(
        lambda: (
            tool.operation_editor.stack.currentWidget() is not None
            and tool.operation_editor.stack.currentWidget().findChild(
                color_widgets._ColorListEditorWidget,
                "figureComposerSetPaletteColorsWidget",
            )
            is not None
        ),
        timeout=1000,
    )


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_figure_composer_set_palette_sequential_editor(qtbot, mode: str) -> None:
    sns = pytest.importorskip("seaborn")
    operation = FigureOperationState.set_palette().model_copy(
        update={"palette_mode": mode}
    )
    tool = _set_palette_tool(qtbot, operation)
    page = _set_palette_page(tool)

    input_combo = page.findChild(
        QtWidgets.QComboBox,
        f"figureComposerSetPalette{mode.title()}InputCombo",
    )
    hue = page.findChild(
        figurecomposer_set_palette._PaletteSliderWidget,
        f"figureComposerSetPalette{mode.title()}HueControl",
    )
    count = page.findChild(
        figurecomposer_set_palette._PaletteSliderWidget,
        f"figureComposerSetPalette{mode.title()}NColorsControl",
    )
    docs_button = page.findChild(
        QtWidgets.QToolButton, "figureComposerSetPaletteDocsButton"
    )
    assert input_combo is not None
    assert input_combo.currentData() == "husl"
    assert [input_combo.itemData(index) for index in range(input_combo.count())] == [
        "husl",
        "hls",
        "rgb",
    ]
    assert hue is not None
    assert hue.spin.minimum() == pytest.approx(0.0)
    assert hue.spin.maximum() == pytest.approx(359.0)
    assert count is not None
    assert count.value() == 10
    assert docs_button is not None
    assert docs_button.property("figure_palette_doc_url") == (
        f"https://seaborn.pydata.org/generated/seaborn.{mode}_palette.html"
    )
    assert (
        page.findChild(QtWidgets.QSpinBox, "figureComposerSetPaletteCountSpin") is None
    )

    hue.spin.setValue(240.0)
    count.spin.setValue(6)
    current = tool.tool_status.operations[0]
    state = current.palette_light if mode == "light" else current.palette_dark
    assert state.color[0] == pytest.approx(240.0)
    assert state.n_colors == 6

    preview = page.findChild(QtWidgets.QWidget, "figureComposerSetPalettePreview")
    assert preview is not None
    swatches = preview.findChildren(
        QtWidgets.QFrame, "figureComposerSetPalettePreviewSwatch"
    )
    expected = getattr(sns, f"{mode}_palette")(
        state.color,
        n_colors=state.n_colors,
        input=state.input,
    )
    assert [swatch.property("palette_color") for swatch in swatches] == [
        QtGui.QColor.fromRgbF(*color).name() for color in expected
    ]


@pytest.mark.parametrize(
    ("mode", "source", "color", "target"),
    [
        ("light", "husl", (220.0, 70.0, 40.0), "rgb"),
        ("dark", "rgb", (0.2, 0.6, 0.8), "hls"),
        ("light", "hls", (0.7, 0.4, 0.6), "husl"),
    ],
)
def test_figure_composer_set_palette_space_switch_preserves_seed_color(
    qtbot,
    mode: str,
    source: str,
    color: tuple[float, float, float],
    target: str,
) -> None:
    sns = pytest.importorskip("seaborn")
    state = FigureSequentialPaletteState(input=source, color=color, n_colors=8)
    operation = FigureOperationState.set_palette().model_copy(
        update={"palette_mode": mode, f"palette_{mode}": state}
    )
    tool = _set_palette_tool(qtbot, operation)
    page = _set_palette_page(tool)
    palette_factory = getattr(sns, f"{mode}_palette")
    before_rgb = palette_factory(
        state.color,
        n_colors=2,
        input=state.input,
    )[-1]

    input_combo = page.findChild(
        QtWidgets.QComboBox,
        f"figureComposerSetPalette{mode.title()}InputCombo",
    )
    assert input_combo is not None
    _activate_combo_data(input_combo, target)

    current = tool.tool_status.operations[0]
    converted = current.palette_light if mode == "light" else current.palette_dark
    assert converted.input == target
    after_rgb = palette_factory(
        converted.color,
        n_colors=2,
        input=converted.input,
    )[-1]
    np.testing.assert_allclose(after_rgb, before_rgb, atol=2e-2)


def test_figure_composer_set_palette_cubehelix_batch_preserves_other_fields(
    qtbot,
) -> None:
    first = FigureOperationState.set_palette(label="first").model_copy(
        update={
            "palette_mode": "cubehelix",
            "palette_cubehelix": FigureCubehelixPaletteState(start=0.1, gamma=0.8),
        }
    )
    second = FigureOperationState.set_palette(label="second").model_copy(
        update={
            "palette_mode": "cubehelix",
            "palette_cubehelix": FigureCubehelixPaletteState(start=1.2, gamma=1.6),
        }
    )
    profile = _figure_composer_profile_source("profile")
    tool = FigureComposerTool(
        profile,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="profile", label="profile"),),
            operations=(first, second),
            primary_source="profile",
        ),
        source_data={"profile": profile},
    )
    qtbot.addWidget(tool)
    _select_operation_rows(tool, (0, 1))
    page = _set_palette_page(tool)

    start_control = page.findChild(
        figurecomposer_set_palette._PaletteSliderWidget,
        "figureComposerSetPaletteCubehelixStartControl",
    )
    assert start_control is not None
    container = start_control.parentWidget()
    assert container is not None
    assert container.findChild(QtWidgets.QLabel, "figureComposerMixedValueMarker")

    start_control.spin.setValue(0.75)
    updated = tool.tool_status.operations
    assert updated[0].palette_cubehelix.start == pytest.approx(0.75)
    assert updated[1].palette_cubehelix.start == pytest.approx(0.75)
    assert updated[0].palette_cubehelix.gamma == pytest.approx(0.8)
    assert updated[1].palette_cubehelix.gamma == pytest.approx(1.6)


def test_figure_composer_set_palette_batch_space_conversion_is_per_step(
    qtbot,
) -> None:
    sns = pytest.importorskip("seaborn")
    first_state = FigureSequentialPaletteState(
        input="husl", color=(30.0, 80.0, 55.0), n_colors=5
    )
    second_state = FigureSequentialPaletteState(
        input="hls", color=(0.1, 0.6, 0.3), n_colors=9
    )
    first = FigureOperationState.set_palette(label="first").model_copy(
        update={"palette_mode": "light", "palette_light": first_state}
    )
    second = FigureOperationState.set_palette(label="second").model_copy(
        update={"palette_mode": "light", "palette_light": second_state}
    )
    profile = _figure_composer_profile_source("profile")
    tool = FigureComposerTool(
        profile,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="profile", label="profile"),),
            operations=(first, second),
            primary_source="profile",
        ),
        source_data={"profile": profile},
    )
    qtbot.addWidget(tool)
    _select_operation_rows(tool, (0, 1))
    page = _set_palette_page(tool)

    input_combo = page.findChild(
        QtWidgets.QComboBox, "figureComposerSetPaletteLightInputCombo"
    )
    assert input_combo is not None
    assert input_combo.currentData() is None
    expected_colors = (
        sns.light_palette(first_state.color, n_colors=2, input=first_state.input)[-1],
        sns.light_palette(second_state.color, n_colors=2, input=second_state.input)[-1],
    )

    _activate_combo_data(input_combo, "hls")
    updated = tool.tool_status.operations
    for index, expected_color in enumerate(expected_colors):
        assert updated[index].palette_light.input == "hls"
        np.testing.assert_allclose(
            sns.light_palette(
                updated[index].palette_light.color,
                n_colors=2,
                input=updated[index].palette_light.input,
            )[-1],
            expected_color,
            atol=2e-2,
        )
    assert updated[0].palette_light.n_colors == 5
    assert updated[1].palette_light.n_colors == 9


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


@pytest.mark.parametrize(
    ("mode", "control_name"),
    [
        ("cubehelix", "figureComposerSetPaletteCubehelixNColorsControl"),
        ("light", "figureComposerSetPaletteLightInputCombo"),
        ("dark", "figureComposerSetPaletteDarkInputCombo"),
    ],
)
def test_figure_composer_generated_palette_editor_disables_without_seaborn(
    qtbot,
    monkeypatch,
    mode: str,
    control_name: str,
) -> None:
    monkeypatch.setattr(figurecomposer_set_palette, "_import_seaborn", lambda: None)
    operation = FigureOperationState.set_palette().model_copy(
        update={"palette_mode": mode}
    )
    tool = _set_palette_tool(qtbot, operation)
    page = _set_palette_page(tool)

    control = page.findChild(QtWidgets.QWidget, control_name)
    message = page.findChild(
        QtWidgets.QLabel, "figureComposerSetPaletteUnavailableLabel"
    )
    assert control is not None
    assert not control.isEnabled()
    assert message is not None
    assert message.property("missing_dependency") == "seaborn"
    assert figurecomposer_set_palette._display_text(tool, operation).startswith(
        "Skipped Set Palette:"
    )

    figurecomposer_rendering._render_into_figure(tool, tool.figure, sync_visible=False)
    assert tool._operation_render_errors == {}


@pytest.mark.parametrize(
    ("mode", "input_space"),
    [
        ("cubehelix", None),
        ("light", "husl"),
        ("light", "hls"),
        ("light", "rgb"),
        ("dark", "husl"),
        ("dark", "hls"),
        ("dark", "rgb"),
    ],
)
def test_figure_composer_generated_palette_render_and_codegen(
    qtbot,
    mode: str,
    input_space: str | None,
) -> None:
    sns = pytest.importorskip("seaborn")
    if mode == "cubehelix":
        cubehelix = FigureCubehelixPaletteState(
            n_colors=4,
            start=0.3,
            rot=-0.4,
            gamma=1.2,
            hue=0.65,
            light=0.9,
            dark=0.1,
            reverse=True,
        )
        palette_operation = FigureOperationState.set_palette().model_copy(
            update={
                "palette_mode": mode,
                "palette_cubehelix": cubehelix,
            }
        )
        expected = sns.cubehelix_palette(
            n_colors=cubehelix.n_colors,
            start=cubehelix.start,
            rot=cubehelix.rot,
            gamma=cubehelix.gamma,
            hue=cubehelix.hue,
            light=cubehelix.light,
            dark=cubehelix.dark,
            reverse=cubehelix.reverse,
        )
    else:
        assert input_space is not None
        colors = {
            "husl": (210.0, 70.0, 45.0),
            "hls": (0.6, 0.4, 0.7),
            "rgb": (0.2, 0.5, 0.8),
        }
        sequential = FigureSequentialPaletteState(
            input=typing.cast("typing.Any", input_space),
            color=colors[input_space],
            n_colors=4,
        )
        palette_operation = FigureOperationState.set_palette().model_copy(
            update={
                "palette_mode": mode,
                f"palette_{mode}": sequential,
            }
        )
        expected = getattr(sns, f"{mode}_palette")(
            sequential.color,
            n_colors=sequential.n_colors,
            input=sequential.input,
        )

    profile = _figure_composer_profile_source("profile")
    line_operation = FigureOperationState.line(label="line", source="profile")
    tool = FigureComposerTool(
        profile,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="profile", label="profile"),),
            operations=(palette_operation, line_operation),
            primary_source="profile",
        ),
        source_data={"profile": profile},
    )
    qtbot.addWidget(tool)

    figurecomposer_rendering._render_into_figure(tool, tool.figure, sync_visible=False)
    assert tool._operation_render_errors == {}
    np.testing.assert_allclose(
        mcolors.to_rgb(tool.figure.axes[0].lines[0].get_color()),
        expected[0],
    )

    code = tool.generated_code()
    assert f"sns.{mode}_palette(" in code
    assert "choose_" not in code
    namespace: dict[str, typing.Any] = {"profile": profile}
    exec(code, namespace)  # noqa: S102
    generated_line = namespace["fig"].axes[0].lines[0]
    np.testing.assert_allclose(
        mcolors.to_rgb(generated_line.get_color()),
        expected[0],
    )


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
