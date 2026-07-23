"""Seaborn palette operation for Figure Composer."""

from __future__ import annotations

import colorsys
import functools
import html
import operator
import typing
import weakref

import matplotlib.pyplot as plt
from qtpy import QtCore, QtGui, QtWidgets

import erlab.interactive.utils
from erlab.interactive._figurecomposer._model._state import (
    FigureOperationKind,
    FigureOperationState,
    FigureSequentialPaletteState,
)
from erlab.interactive._figurecomposer._operations._base import (
    AddStepActionSpec,
    OperationSpec,
    _always_render_cache_safe,
    _empty_source_editor,
    _no_invalid_target,
    _uses_no_source_section,
)
from erlab.interactive._figurecomposer._ui._color_widgets import (
    _ColorListEditorWidget,
    _ColorPickerButton,
)
from erlab.interactive._figurecomposer._ui._operation_editor import StepSection
from erlab.plotting.colors import close_to_white

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from matplotlib.figure import Figure

    from erlab.interactive._figurecomposer._tool import FigureComposerTool
    from erlab.interactive._figurecomposer._ui._operation_editor import (
        FigureOperationEditor,
    )

_SEABORN_DOC_ROOT = "https://seaborn.pydata.org/generated/seaborn"
_SET_PALETTE_DOC_URL = f"{_SEABORN_DOC_ROOT}.set_palette.html"
_PALETTE_DOC_URLS = {
    "named": _SET_PALETTE_DOC_URL,
    "colors": _SET_PALETTE_DOC_URL,
    "cubehelix": f"{_SEABORN_DOC_ROOT}.cubehelix_palette.html",
    "diverging": f"{_SEABORN_DOC_ROOT}.diverging_palette.html",
    "light": f"{_SEABORN_DOC_ROOT}.light_palette.html",
    "dark": f"{_SEABORN_DOC_ROOT}.dark_palette.html",
}
_SEABORN_NAMED_PALETTES = (
    "deep",
    "deep6",
    "muted",
    "muted6",
    "pastel",
    "pastel6",
    "bright",
    "bright6",
    "dark",
    "dark6",
    "colorblind",
    "colorblind6",
)
_COLORBREWER_QUALITATIVE_PALETTES = (
    "tab10",
    "tab20",
    "tab20b",
    "tab20c",
    "Set1",
    "Set2",
    "Set3",
    "Accent",
    "Paired",
    "Pastel1",
    "Pastel2",
    "Dark2",
)
_SEABORN_CONTINUOUS_PALETTES = (
    "rocket",
    "rocket_r",
    "mako",
    "mako_r",
    "flare",
    "flare_r",
    "crest",
    "crest_r",
    "icefire",
    "icefire_r",
    "vlag",
    "vlag_r",
)
_SEABORN_SPECIAL_PALETTES = (
    "hls",
    "husl",
    "ch:s=.25,rot=-.25",
    "light:#5A9",
    "dark:#5A9_r",
    "blend:#7AB,#EDA",
)
_HIDDEN_PALETTE_OPTIONS = frozenset({"jet", "jet_r"})
_DESAT_AUTO_VALUE = -0.01
_PALETTE_COUNT_MAX = 256
_PALETTE_ICON_SIZE = (72, 14)
_PALETTE_ICON_COLORS = 8
_PALETTE_MODE_LABELS = {
    "named": "Named palette",
    "colors": "Custom colors",
    "cubehelix": "Cubehelix palette",
    "diverging": "Diverging palette",
    "light": "Light palette",
    "dark": "Dark palette",
}
_GENERATED_PALETTE_MODES = frozenset({"cubehelix", "diverging", "light", "dark"})
_SEQUENTIAL_PALETTE_INPUT_LABELS = {
    "husl": "HUSL",
    "hls": "HLS",
    "rgb": "RGB",
}


def _palette_tooltip_text_color(qcolor: QtGui.QColor) -> str:
    color = (qcolor.redF(), qcolor.greenF(), qcolor.blueF())
    return "#000000" if close_to_white(color) else "#ffffff"


def _palette_tooltip_html(hex_color: str, text_color: str) -> str:
    font_family = QtGui.QFontDatabase.systemFont(
        QtGui.QFontDatabase.SystemFont.FixedFont
    ).family()
    escaped_family = html.escape(font_family, quote=True)
    escaped_hex = html.escape(hex_color)
    return (
        '<qt><span style="'
        f"font-family: '{escaped_family}', monospace; "
        f"color: {text_color}; "
        f"background-color: {escaped_hex}; "
        "padding: 2px 4px;"
        f'">{escaped_hex}</span></qt>'
    )


class _PaletteSwatch(QtWidgets.QFrame):
    """Palette preview swatch with copyable hex color."""

    def __init__(
        self,
        qcolor: QtGui.QColor,
        index: int,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        hex_color = qcolor.name()
        self._hex_color = hex_color
        self.setObjectName("figureComposerSetPalettePreviewSwatch")
        self.setProperty("palette_color_index", index)
        self.setProperty("palette_color", hex_color)
        text_color = _palette_tooltip_text_color(qcolor)
        self.setProperty("palette_tooltip_text_color", text_color)
        self.setProperty("palette_tooltip_font_family", "monospace")
        self.setAccessibleName(f"Palette color {index + 1}: {hex_color}")
        self.setToolTip(_palette_tooltip_html(hex_color, text_color))
        self.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.setMinimumSize(18, 18)
        self.setMaximumSize(28, 18)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        border = self.palette().color(QtGui.QPalette.ColorRole.Mid).name()
        self.setStyleSheet(
            f"background-color: {hex_color}; border: 1px solid {border};"
        )

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent | None) -> None:
        if event is None:
            return
        menu = QtWidgets.QMenu(self)
        copy_action = menu.addAction("Copy Hex Code")
        if copy_action is None:
            event.accept()
            return
        copy_action.setData(self._hex_color)
        chosen = menu.exec(event.globalPos())
        if chosen is copy_action:
            self.copy_hex_to_clipboard()
        event.accept()

    def copy_hex_to_clipboard(self) -> None:
        clipboard = typing.cast("QtGui.QClipboard", QtWidgets.QApplication.clipboard())
        clipboard.setText(self._hex_color)


class _PalettePreviewWidget(QtWidgets.QWidget):
    """Compact swatch preview for the resolved seaborn palette."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("figureComposerSetPalettePreview")
        self.setProperty("palette_preview", True)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)

    def set_colors(self, colors: Sequence[typing.Any]) -> None:
        layout = typing.cast("QtWidgets.QHBoxLayout", self.layout())
        while layout.count():
            item = layout.takeAt(0)
            if item is not None and (widget := item.widget()) is not None:
                widget.setParent(None)
                widget.deleteLater()
        if not colors:
            label = QtWidgets.QLabel("No preview", self)
            label.setObjectName("figureComposerSetPalettePreviewUnavailable")
            label.setEnabled(False)
            label.setToolTip("No colors could be resolved for this palette.")
            layout.addWidget(label)
            layout.addStretch(1)
            return
        for index, color in enumerate(colors):
            qcolor = _qt_color(color)
            swatch = _PaletteSwatch(qcolor, index, self)
            layout.addWidget(swatch, 1)
        layout.addStretch(1)


class _PaletteSliderWidget(QtWidgets.QWidget):
    """Synchronized slider and spinbox for one chooser parameter."""

    valueChanged = QtCore.Signal(object)
    previewValueChanged = QtCore.Signal(object)

    def __init__(
        self,
        minimum: float,
        maximum: float,
        value: float,
        *,
        decimals: int = 2,
        single_step: float = 0.1,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._scale = 10**decimals

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, self)
        self.slider.setRange(round(minimum * self._scale), round(maximum * self._scale))
        self.slider.setSingleStep(max(1, round(single_step * self._scale)))
        self.slider.setMinimumWidth(90)

        self.spin = QtWidgets.QDoubleSpinBox(self)
        self.spin.setRange(minimum, maximum)
        self.spin.setDecimals(decimals)
        self.spin.setSingleStep(single_step)
        self.spin.setKeyboardTracking(False)

        layout.addWidget(self.slider, 1)
        layout.addWidget(self.spin)
        self.setValue(value)

    def value(self) -> float:
        return self.spin.value()

    def setValue(self, value: float) -> None:
        with QtCore.QSignalBlocker(self.spin), QtCore.QSignalBlocker(self.slider):
            self.spin.setValue(value)
            self.slider.setValue(round(value * self._scale))

    def _slider_changed(self, value: int) -> None:
        self.spin.setValue(value / self._scale)

    def _spin_changed(self, _value: float) -> None:
        with QtCore.QSignalBlocker(self.slider):
            self.slider.setValue(round(float(self.spin.value()) * self._scale))
        signal = (
            self.previewValueChanged
            if self.slider.isSliderDown()
            else self.valueChanged
        )
        signal.emit(self.value())

    def _slider_released(self) -> None:
        self.valueChanged.emit(self.value())


def _qt_color(color: typing.Any) -> QtGui.QColor:
    red, green, blue = color[:3]
    return QtGui.QColor.fromRgbF(float(red), float(green), float(blue))


def _import_seaborn() -> typing.Any | None:
    try:
        import seaborn as sns
    except ImportError:
        return None
    return sns


def _palette_options(
    operation: FigureOperationState, sns: typing.Any | None = None
) -> tuple[str, ...]:
    options: list[str] = []
    if sns is not None:
        options.extend(getattr(sns.palettes, "QUAL_PALETTES", ()))
    options.extend(
        (
            *_SEABORN_NAMED_PALETTES,
            *_COLORBREWER_QUALITATIVE_PALETTES,
            *_SEABORN_CONTINUOUS_PALETTES,
            *_SEABORN_SPECIAL_PALETTES,
            *plt.colormaps(),
        )
    )
    options = [option for option in options if option not in _HIDDEN_PALETTE_OPTIONS]
    if (
        operation.palette_name not in options
        and operation.palette_name not in _HIDDEN_PALETTE_OPTIONS
    ):
        options.append(operation.palette_name)
    return tuple(dict.fromkeys(options))


def _palette_display_text(operation: FigureOperationState) -> str:
    if operation.palette_mode == "colors":
        count = len(operation.palette_colors)
        return f"Custom colors ({count})" if count else "Custom colors"
    if operation.palette_mode == "cubehelix":
        return f"Cubehelix palette ({operation.palette_cubehelix.n_colors})"
    if operation.palette_mode == "diverging":
        state = operation.palette_diverging
        return f"Diverging palette ({state.center}, {state.n_colors})"
    if operation.palette_mode in {"light", "dark"}:
        state = getattr(operation, f"palette_{operation.palette_mode}")
        label = _PALETTE_MODE_LABELS[operation.palette_mode]
        return f"{label} ({state.input.upper()}, {state.n_colors})"
    return operation.palette_name


def _palette_value(operation: FigureOperationState) -> str | tuple[str, ...]:
    if operation.palette_mode == "colors":
        return operation.palette_colors
    return operation.palette_name


def _palette_has_effect(operation: FigureOperationState) -> bool:
    return operation.palette_mode != "colors" or bool(operation.palette_colors)


def _palette_call_kwargs(operation: FigureOperationState) -> dict[str, typing.Any]:
    kwargs: dict[str, typing.Any] = {}
    if operation.palette_n_colors is not None:
        kwargs["n_colors"] = operation.palette_n_colors
    if operation.palette_desat is not None:
        kwargs["desat"] = operation.palette_desat
    if operation.palette_color_codes:
        kwargs["color_codes"] = True
    return kwargs


def _palette_colors(
    sns: typing.Any | None,
    palette: str | tuple[str, ...],
    n_colors: int | None,
    desat: float | None,
) -> tuple[typing.Any, ...]:
    if sns is None:
        return ()
    try:
        return tuple(sns.color_palette(palette, n_colors=n_colors, desat=desat))
    except (TypeError, ValueError):
        return ()


def _generated_palette_colors(
    sns: typing.Any | None, operation: FigureOperationState
) -> tuple[typing.Any, ...]:
    if sns is None:
        return ()
    if operation.palette_mode == "cubehelix":
        state = operation.palette_cubehelix
        return tuple(
            sns.cubehelix_palette(
                n_colors=state.n_colors,
                start=state.start,
                rot=state.rot,
                gamma=state.gamma,
                hue=state.hue,
                light=state.light,
                dark=state.dark,
                reverse=state.reverse,
            )
        )
    if operation.palette_mode == "diverging":
        state = operation.palette_diverging
        return tuple(
            sns.diverging_palette(
                h_neg=state.h_neg,
                h_pos=state.h_pos,
                s=state.s,
                l=state.l,
                sep=state.sep,
                n=state.n_colors,
                center=state.center,
            )
        )
    if operation.palette_mode == "light":
        state = operation.palette_light
        palette_factory = sns.light_palette
    else:
        state = operation.palette_dark
        palette_factory = sns.dark_palette
    return tuple(
        palette_factory(
            state.color,
            n_colors=state.n_colors,
            input=state.input,
        )
    )


def _resolved_palette_colors(
    sns: typing.Any | None, operation: FigureOperationState
) -> tuple[typing.Any, ...]:
    if operation.palette_mode in _GENERATED_PALETTE_MODES:
        return _generated_palette_colors(sns, operation)
    return _palette_colors(
        sns,
        _palette_value(operation),
        operation.palette_n_colors,
        operation.palette_desat,
    )


def _convert_palette_color(
    sns: typing.Any,
    color: tuple[float, float, float],
    source: str,
    target: str,
) -> tuple[float, float, float]:
    if source == target:
        return color
    if source == "husl":
        rgb = tuple(float(value) for value in sns.external.husl.husl_to_rgb(*color))
    elif source == "hls":
        rgb = colorsys.hls_to_rgb(*color)
    else:
        rgb = color

    if target == "husl":
        converted = tuple(float(value) for value in sns.external.husl.rgb_to_husl(*rgb))
    elif target == "hls":
        converted = colorsys.rgb_to_hls(*rgb)
    else:
        converted = rgb
    limits = (
        ((0.0, 359.0), (0.0, 99.0), (0.0, 99.0))
        if target == "husl"
        else ((0.0, 1.0),) * 3
    )
    return typing.cast(
        "tuple[float, float, float]",
        tuple(
            min(max(value, lower), upper)
            for value, (lower, upper) in zip(converted, limits, strict=True)
        ),
    )


def _palette_hex_colors(colors: Sequence[typing.Any]) -> tuple[str, ...]:
    return tuple(_qt_color(color).name() for color in colors)


def _palette_icon(
    sns: typing.Any | None,
    palette_name: str,
) -> QtGui.QIcon:
    colors = _palette_hex_colors(
        _palette_colors(sns, palette_name, _PALETTE_ICON_COLORS, None)
    )
    if not colors:
        return QtGui.QIcon()
    return QtGui.QIcon(_palette_pixmap(colors))


@functools.lru_cache(maxsize=512)
def _palette_pixmap(
    hex_colors: tuple[str, ...],
    width: int = _PALETTE_ICON_SIZE[0],
    height: int = _PALETTE_ICON_SIZE[1],
) -> QtGui.QPixmap:
    pixmap = QtGui.QPixmap(width, height)
    pixmap.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(pixmap)
    try:
        color_count = len(hex_colors)
        for index, color in enumerate(hex_colors):
            left = round(index * width / color_count)
            right = round((index + 1) * width / color_count)
            painter.fillRect(
                QtCore.QRect(left, 0, max(1, right - left), height),
                QtGui.QColor(color),
            )
        painter.setPen(QtGui.QColor("#808080"))
        painter.drawRect(0, 0, width - 1, height - 1)
    finally:
        painter.end()
    return pixmap


def _apply_palette_combo_icons(
    combo: QtWidgets.QComboBox,
    sns: typing.Any | None,
) -> None:
    combo.setIconSize(QtCore.QSize(*_PALETTE_ICON_SIZE))
    for index in range(combo.count()):
        icon = _palette_icon(sns, combo.itemText(index))
        if not icon.isNull():
            combo.setItemIcon(index, icon)


def _apply_palette_to_existing_axes(fig: Figure, sns: typing.Any) -> None:
    colors = sns.color_palette()
    if not colors:
        return
    for axis in fig.axes:
        axis.set_prop_cycle(color=colors)


def _render_set_palette(
    _tool: FigureComposerTool,
    operation: FigureOperationState,
    fig: Figure,
    _axs: typing.Any,
) -> None:
    sns = _import_seaborn()
    if sns is None or not _palette_has_effect(operation):
        return
    if operation.palette_mode in _GENERATED_PALETTE_MODES:
        sns.set_palette(_generated_palette_colors(sns, operation))
    else:
        sns.set_palette(_palette_value(operation), **_palette_call_kwargs(operation))
    _apply_palette_to_existing_axes(fig, sns)


def _create_operation(_tool: FigureComposerTool) -> FigureOperationState:
    return FigureOperationState.set_palette(label="set palette")


def _display_text(_tool: FigureComposerTool, operation: FigureOperationState) -> str:
    if _import_seaborn() is None:
        return f"Skipped Set Palette: {_palette_display_text(operation)}"
    return f"Set Palette: {_palette_display_text(operation)}"


def _tooltip(_tool: FigureComposerTool, operation: FigureOperationState) -> str:
    if _import_seaborn() is None:
        return "Install seaborn to apply this palette step.\nTargets: figure"
    return (
        "Sets the Matplotlib line color cycle with seaborn.set_palette().\n"
        f"Palette: {_palette_display_text(operation)}\n"
        "Targets: figure"
    )


def _target_text(_tool: FigureComposerTool, _operation: FigureOperationState) -> str:
    return "Figure"


def _add_palette_slider(
    editor: FigureOperationEditor,
    layout: QtWidgets.QFormLayout,
    page: QtWidgets.QWidget,
    *,
    label: str,
    object_name: str,
    value: float,
    minimum: float,
    maximum: float,
    changed: Callable[[float, bool], None],
    tooltip: str,
    mixed: bool,
    enabled: bool,
    decimals: int = 2,
    single_step: float = 0.1,
) -> _PaletteSliderWidget:
    slider = _PaletteSliderWidget(
        minimum,
        maximum,
        value,
        decimals=decimals,
        single_step=single_step,
        parent=page,
    )
    slider.setObjectName(object_name)
    slider.slider.setObjectName(f"{object_name}Slider")
    slider.spin.setObjectName(f"{object_name}Spin")
    slider.setAccessibleName(label)
    slider.slider.setAccessibleName(label)
    slider.spin.setAccessibleName(label)
    slider.setEnabled(enabled)
    editor.connect_signal(
        slider.slider, slider.slider.valueChanged, slider._slider_changed
    )
    editor.connect_signal(slider.spin, slider.spin.valueChanged, slider._spin_changed)
    editor.connect_signal(
        slider.slider, slider.slider.sliderReleased, slider._slider_released
    )
    editor.connect_signal(
        slider,
        slider.previewValueChanged,
        lambda value: changed(float(value), True),
    )
    editor.connect_signal(
        slider,
        slider.valueChanged,
        lambda value: changed(float(value), False),
    )
    editor.add_form_row(
        layout,
        label,
        editor.mixed_value_widget(slider, mixed=mixed, parent=page),
        tooltip,
    )
    return slider


def _editor_sections(
    editor: FigureOperationEditor, operation: FigureOperationState
) -> tuple[StepSection, ...]:
    page, layout = editor.new_form_page("figureComposerSetPalettePage")
    sns = _import_seaborn()
    available = sns is not None
    editor.add_form_section(
        layout,
        "Palette",
        object_name="figureComposerSetPaletteSection",
    )

    preview = _PalettePreviewWidget(page)
    preview.setEnabled(available)
    preview_operation = [operation]
    seed_color_button_ref: weakref.ReferenceType[_ColorPickerButton] | None = None
    diverging_color_button_refs: dict[
        str, weakref.ReferenceType[_ColorPickerButton]
    ] = {}

    def refresh_preview() -> None:
        current = preview_operation[0]
        preview.set_colors(_resolved_palette_colors(sns, current))
        seed_color_button = (
            seed_color_button_ref() if seed_color_button_ref is not None else None
        )
        if (
            seed_color_button is not None
            and sns is not None
            and current.palette_mode in {"light", "dark"}
            and erlab.interactive.utils.qt_is_valid(seed_color_button)
        ):
            state = typing.cast(
                "FigureSequentialPaletteState",
                getattr(current, f"palette_{current.palette_mode}"),
            )
            seed_color_button.setColor(
                _qt_color(
                    _convert_palette_color(
                        sns,
                        state.color,
                        state.input,
                        "rgb",
                    )
                )
            )
        if (
            diverging_color_button_refs
            and sns is not None
            and current.palette_mode == "diverging"
        ):
            state = current.palette_diverging
            for field, button_ref in diverging_color_button_refs.items():
                button = button_ref()
                if button is None or not erlab.interactive.utils.qt_is_valid(button):
                    continue
                button.setColor(
                    _qt_color(
                        _convert_palette_color(
                            sns,
                            (getattr(state, field), state.s, state.l),
                            "husl",
                            "rgb",
                        )
                    )
                )

    def update_operation(**updates: typing.Any) -> None:
        preview_operation[0] = preview_operation[0].model_copy(update=updates)
        refresh_preview()
        editor.request_update(**updates)

    def transform_palette_state(
        attribute: str,
        updater: Callable[[typing.Any], typing.Any],
        *,
        preview_only: bool = False,
        rebuild: bool = False,
    ) -> None:
        def transform(
            _index: int, target: FigureOperationState
        ) -> FigureOperationState:
            return target.model_copy(
                update={attribute: updater(getattr(target, attribute))}
            )

        preview_operation[0] = transform(0, preview_operation[0])
        refresh_preview()
        if preview_only:
            return
        editor.request_transform(
            transform,
            rebuild_editor=rebuild,
            defer_editor_rebuild=rebuild,
        )

    if not available:
        message = QtWidgets.QLabel(
            "Install seaborn to preview or apply palettes.", page
        )
        message.setObjectName("figureComposerSetPaletteUnavailableLabel")
        message.setProperty("missing_dependency", "seaborn")
        message.setWordWrap(True)
        editor.add_form_row(
            layout,
            "Dependency",
            message,
            "This step needs seaborn in the active Python environment.",
        )

    def update_palette_mode(mode: str) -> None:
        def transform(
            _index: int, target: FigureOperationState
        ) -> FigureOperationState:
            updates: dict[str, typing.Any] = {"palette_mode": mode}
            if mode == "colors" and not target.palette_colors:
                seeded_colors = _palette_hex_colors(
                    _resolved_palette_colors(sns, target)
                )
                if seeded_colors:
                    updates["palette_colors"] = seeded_colors
            return target.model_copy(update=updates)

        preview_operation[0] = transform(0, preview_operation[0])
        refresh_preview()
        editor.request_transform(
            transform,
            rebuild_editor=True,
            defer_editor_rebuild=True,
        )

    mode_mixed = editor.batch_is_mixed(operation, lambda target: target.palette_mode)
    mode_combo = QtWidgets.QComboBox(page)
    mode_combo.setObjectName("figureComposerSetPaletteModeCombo")
    for mode, text in _PALETTE_MODE_LABELS.items():
        mode_combo.addItem(text, mode)
    if mode_mixed:
        mode_combo.insertItem(0, "(multiple values)", None)
        item = typing.cast("typing.Any", mode_combo.model()).item(0)
        if item is not None:
            item.setEnabled(False)
        mode_combo.setCurrentIndex(0)
    else:
        mode_combo.setCurrentIndex(max(mode_combo.findData(operation.palette_mode), 0))
    mode_combo.setEnabled(available)
    mode_combo.setToolTip(
        "Choose a named palette, explicit colors, or a generated sequential palette."
    )
    editor.connect_signal(
        mode_combo,
        mode_combo.activated,
        lambda _index, combo=mode_combo: (
            None
            if combo.currentData() is None
            else update_palette_mode(str(combo.currentData()))
        ),
    )

    mode_row = QtWidgets.QWidget(page)
    mode_row.setObjectName("figureComposerSetPaletteModeRow")
    mode_layout = QtWidgets.QHBoxLayout(mode_row)
    mode_layout.setContentsMargins(0, 0, 0, 0)
    mode_layout.setSpacing(6)
    mode_layout.addWidget(mode_combo, 1)
    docs_button = QtWidgets.QToolButton(mode_row)
    docs_button.setObjectName("figureComposerSetPaletteDocsButton")
    docs_button.setText("Docs")
    docs_button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextOnly)
    docs_mode = "named" if mode_mixed else operation.palette_mode
    docs_url = _PALETTE_DOC_URLS[docs_mode]
    docs_button.setProperty("figure_palette_doc_url", docs_url)
    docs_button.setToolTip("Open documentation for this seaborn palette function.")

    def open_docs(_checked: bool = False) -> None:
        QtGui.QDesktopServices.openUrl(QtCore.QUrl(docs_url))

    editor.connect_signal(
        docs_button,
        docs_button.clicked,
        open_docs,
    )
    mode_layout.addWidget(docs_button)
    editor.add_form_row(
        layout,
        "Use",
        mode_row,
        "Choose how this step constructs the line color palette.",
    )

    if not mode_mixed and operation.palette_mode == "colors":

        def update_palette_colors(colors: typing.Any) -> None:
            colors = tuple(str(color).strip() for color in colors if str(color).strip())
            update_operation(
                palette_mode="colors",
                palette_colors=colors,
            )

        colors_mixed = editor.batch_is_mixed(
            operation, lambda target: target.palette_colors
        )
        colors_widget = _ColorListEditorWidget(
            () if colors_mixed else operation.palette_colors,
            parent=page,
        )
        colors_widget.setObjectName("figureComposerSetPaletteColorsWidget")
        colors_widget.setMainEditObjectName("figureComposerSetPaletteColorsEdit")
        if colors_mixed:
            colors_widget.setMixedPlaceholder("(multiple values)")
        colors_widget.setEnabled(available)
        colors_widget.setToolTip("Colors passed as a sequence to seaborn.set_palette.")
        editor.connect_value_signal(
            colors_widget,
            colors_widget.colorsChanged,
            lambda colors: tuple(colors),
            update_palette_colors,
            unchanged_mixed=colors_widget.batchUnchanged,
        )
        editor.add_form_row(
            layout,
            "Colors",
            colors_widget,
            "Explicit color sequence passed to seaborn.set_palette.",
        )
    elif not mode_mixed and operation.palette_mode == "named":
        palette_mixed = editor.batch_is_mixed(
            operation, lambda target: target.palette_name
        )
        palette_combo = editor.combo(
            _palette_options(operation, sns),
            None if palette_mixed else operation.palette_name,
            lambda text: update_operation(palette_name=text),
            parent=page,
            mixed=palette_mixed,
            enabled=available,
        )
        palette_combo.setObjectName("figureComposerSetPaletteNameCombo")
        palette_combo.setToolTip(
            "Palette name passed to seaborn.set_palette. The list includes seaborn "
            "palettes and Matplotlib colormaps."
        )
        _apply_palette_combo_icons(palette_combo, sns)
        editor.add_form_row(
            layout,
            "Palette",
            palette_combo,
            "Named seaborn or Matplotlib palette passed to seaborn.set_palette.",
        )

    if not mode_mixed and operation.palette_mode in {"named", "colors"}:
        count_mixed = editor.batch_is_mixed(
            operation, lambda target: target.palette_n_colors
        )
        count_spin = QtWidgets.QSpinBox(page)
        count_spin.setObjectName("figureComposerSetPaletteCountSpin")
        count_spin.setRange(0, _PALETTE_COUNT_MAX)
        count_spin.setSpecialValueText("Auto")
        count_spin.setValue(operation.palette_n_colors or 0)
        count_spin.setEnabled(available)

        def update_count(value: typing.Any) -> None:
            update_operation(palette_n_colors=None if value == 0 else int(value))

        editor.connect_value_signal(
            count_spin,
            count_spin.valueChanged,
            lambda *_args: count_spin.value(),
            update_count,
        )

        desat_mixed = editor.batch_is_mixed(
            operation, lambda target: target.palette_desat
        )
        desat_spin = QtWidgets.QDoubleSpinBox(page)
        desat_spin.setObjectName("figureComposerSetPaletteSaturationSpin")
        desat_spin.setRange(_DESAT_AUTO_VALUE, 1.0)
        desat_spin.setDecimals(2)
        desat_spin.setSingleStep(0.05)
        desat_spin.setSpecialValueText("Auto")
        desat_spin.setValue(
            _DESAT_AUTO_VALUE
            if operation.palette_desat is None
            else operation.palette_desat
        )
        desat_spin.setEnabled(available)

        def update_desat(value: typing.Any) -> None:
            update_operation(
                palette_desat=(None if value <= _DESAT_AUTO_VALUE else float(value))
            )

        editor.connect_value_signal(
            desat_spin,
            desat_spin.valueChanged,
            lambda *_args: desat_spin.value(),
            update_desat,
        )

        editor.add_compound_form_row(
            layout,
            "Options",
            (
                (
                    "Colors",
                    editor.mixed_value_widget(
                        count_spin, mixed=count_mixed, parent=page
                    ),
                    "Number of colors requested from the palette. Auto uses "
                    "seaborn's default length.",
                ),
                (
                    "Saturation",
                    editor.mixed_value_widget(
                        desat_spin, mixed=desat_mixed, parent=page
                    ),
                    "Scale palette saturation. Auto uses seaborn's default saturation.",
                ),
            ),
            "Optional arguments passed to seaborn.set_palette.",
        )

        color_codes_mixed = editor.batch_is_mixed(
            operation, lambda target: target.palette_color_codes
        )
        color_codes_check = editor.check_box(
            operation.palette_color_codes,
            lambda checked: update_operation(palette_color_codes=checked),
            parent=page,
            mixed=color_codes_mixed,
        )
        color_codes_check.setObjectName("figureComposerSetPaletteColorCodesCheck")
        color_codes_check.setEnabled(available)
        editor.add_form_row(
            layout,
            "Color codes",
            color_codes_check,
            "Map matplotlib color-code letters such as b, g, and r to this palette.",
        )

    elif not mode_mixed and operation.palette_mode == "cubehelix":
        state = operation.palette_cubehelix
        count_mixed = editor.batch_is_mixed(
            operation, lambda target: target.palette_cubehelix.n_colors
        )
        count_spin = QtWidgets.QSpinBox(page)
        count_spin.setObjectName("figureComposerSetPaletteCubehelixNColorsSpin")
        count_spin.setAccessibleName("Colors")
        count_spin.setRange(2, 2_147_483_647)
        count_spin.setValue(state.n_colors)
        count_spin.setKeyboardTracking(False)
        count_spin.setEnabled(available)
        editor.connect_value_signal(
            count_spin,
            count_spin.valueChanged,
            lambda *_args: count_spin.value(),
            lambda value: transform_palette_state(
                "palette_cubehelix",
                lambda current: current.model_copy(update={"n_colors": int(value)}),
            ),
        )
        editor.add_form_row(
            layout,
            "Colors",
            editor.mixed_value_widget(count_spin, mixed=count_mixed, parent=page),
            "Number of colors in the generated cubehelix palette.",
        )

        cubehelix_controls = (
            (
                "start",
                "Start",
                "Start",
                0.0,
                3.0,
                2,
                0.1,
                "Starting hue for the cubehelix rotation.",
            ),
            (
                "rot",
                "Rotation",
                "Rotation",
                -1.0,
                1.0,
                2,
                0.1,
                "Number and direction of rotations through the hue space.",
            ),
            (
                "gamma",
                "Gamma",
                "Gamma",
                0.0,
                5.0,
                2,
                0.1,
                "Gamma correction applied to the intensity ramp.",
            ),
            (
                "hue",
                "Hue",
                "Hue",
                0.0,
                1.0,
                2,
                0.1,
                "Saturation of the cubehelix colors.",
            ),
            (
                "light",
                "Light",
                "Light",
                0.0,
                1.0,
                2,
                0.1,
                "Intensity of the light end of the palette.",
            ),
            (
                "dark",
                "Dark",
                "Dark",
                0.0,
                1.0,
                2,
                0.1,
                "Intensity of the dark end of the palette.",
            ),
        )
        for (
            field,
            label,
            object_suffix,
            minimum,
            maximum,
            decimals,
            single_step,
            tooltip,
        ) in cubehelix_controls:
            mixed = editor.batch_is_mixed(
                operation,
                operator.attrgetter(f"palette_cubehelix.{field}"),
            )

            def update_cubehelix(
                value: float,
                preview_only: bool,
                *,
                field: str = field,
            ) -> None:
                transform_palette_state(
                    "palette_cubehelix",
                    lambda current: current.model_copy(update={field: float(value)}),
                    preview_only=preview_only,
                )

            _add_palette_slider(
                editor,
                layout,
                page,
                label=label,
                object_name=(
                    f"figureComposerSetPaletteCubehelix{object_suffix}Control"
                ),
                value=float(getattr(state, field)),
                minimum=minimum,
                maximum=maximum,
                changed=update_cubehelix,
                tooltip=tooltip,
                mixed=mixed,
                enabled=available,
                decimals=decimals,
                single_step=single_step,
            )

        reverse_mixed = editor.batch_is_mixed(
            operation, lambda target: target.palette_cubehelix.reverse
        )
        reverse_check = editor.check_box(
            state.reverse,
            lambda checked: transform_palette_state(
                "palette_cubehelix",
                lambda current: current.model_copy(update={"reverse": checked}),
            ),
            parent=page,
            mixed=reverse_mixed,
        )
        reverse_check.setObjectName("figureComposerSetPaletteCubehelixReverseCheck")
        reverse_check.setEnabled(available)
        editor.add_form_row(
            layout,
            "Reverse",
            reverse_check,
            "Reverse the generated cubehelix palette.",
        )

    elif not mode_mixed and operation.palette_mode == "diverging":
        state = operation.palette_diverging
        diverging_controls = (
            (
                "h_neg",
                "Negative hue",
                "NegativeHue",
                0.0,
                359.0,
                "Hue for the negative end of the palette.",
            ),
            (
                "h_pos",
                "Positive hue",
                "PositiveHue",
                0.0,
                359.0,
                "Hue for the positive end of the palette.",
            ),
            (
                "s",
                "Saturation",
                "Saturation",
                0.0,
                99.0,
                "Shared saturation of the two palette endpoints.",
            ),
            (
                "l",
                "Lightness",
                "Lightness",
                0.0,
                99.0,
                "Shared lightness of the two palette endpoints.",
            ),
            (
                "sep",
                "Separation",
                "Separation",
                1.0,
                50.0,
                "Size of the intermediate region around the center.",
            ),
        )
        for (
            field,
            label,
            object_suffix,
            minimum,
            maximum,
            tooltip,
        ) in diverging_controls:
            mixed = editor.batch_is_mixed(
                operation,
                operator.attrgetter(f"palette_diverging.{field}"),
            )

            def update_diverging(
                value: float,
                preview_only: bool,
                *,
                field: str = field,
            ) -> None:
                normalized: float | int = int(value) if field == "sep" else float(value)
                transform_palette_state(
                    "palette_diverging",
                    lambda current: current.model_copy(update={field: normalized}),
                    preview_only=preview_only,
                )

            _add_palette_slider(
                editor,
                layout,
                page,
                label=label,
                object_name=(
                    f"figureComposerSetPaletteDiverging{object_suffix}Control"
                ),
                value=float(getattr(state, field)),
                minimum=minimum,
                maximum=maximum,
                changed=update_diverging,
                tooltip=tooltip,
                mixed=mixed,
                enabled=available,
                decimals=0,
                single_step=1.0,
            )

            if field not in {"h_neg", "h_pos"}:
                continue
            color_mixed = editor.batch_is_mixed(
                operation,
                typing.cast(
                    "Callable[[FigureOperationState], tuple[float, float, float]]",
                    lambda target, field=field: (
                        getattr(target.palette_diverging, field),
                        target.palette_diverging.s,
                        target.palette_diverging.l,
                    ),
                ),
            )
            endpoint_rgb = (
                _convert_palette_color(
                    sns,
                    (getattr(state, field), state.s, state.l),
                    "husl",
                    "rgb",
                )
                if sns is not None
                else (0.0, 0.0, 0.0)
            )
            color_button = _ColorPickerButton(
                page,
                color=_qt_color(endpoint_rgb),
                show_alpha_channel=False,
            )
            endpoint = "Negative" if field == "h_neg" else "Positive"
            color_button.setObjectName(
                f"figureComposerSetPaletteDiverging{endpoint}ColorButton"
            )
            color_button.setAccessibleName(f"Choose {endpoint.lower()} endpoint color")
            color_button.setToolTip(f"Choose the {endpoint.lower()} endpoint color.")
            color_button.setEnabled(available)

            def update_endpoint_color(
                color: QtGui.QColor,
                *,
                field: str = field,
            ) -> None:
                husl = _convert_palette_color(
                    sns,
                    (color.redF(), color.greenF(), color.blueF()),
                    "rgb",
                    "husl",
                )
                husl = tuple(float(round(value)) for value in husl)
                transform_palette_state(
                    "palette_diverging",
                    lambda current: current.model_copy(
                        update={field: husl[0], "s": husl[1], "l": husl[2]}
                    ),
                    rebuild=True,
                )

            editor.connect_signal(
                color_button,
                color_button.colorSelected,
                update_endpoint_color,
            )
            diverging_color_button_refs[field] = weakref.ref(color_button)
            editor.add_form_row(
                layout,
                f"{endpoint} color",
                editor.mixed_value_widget(
                    color_button,
                    mixed=color_mixed,
                    parent=page,
                ),
                (
                    f"Choose the {endpoint.lower()} endpoint with the system color "
                    "picker; saturation and lightness are shared by both endpoints."
                ),
            )

        count_mixed = editor.batch_is_mixed(
            operation, lambda target: target.palette_diverging.n_colors
        )
        count_spin = QtWidgets.QSpinBox(page)
        count_spin.setObjectName("figureComposerSetPaletteDivergingNColorsSpin")
        count_spin.setAccessibleName("Colors")
        count_spin.setRange(2, 2_147_483_647)
        count_spin.setValue(state.n_colors)
        count_spin.setKeyboardTracking(False)
        count_spin.setEnabled(available)
        editor.connect_value_signal(
            count_spin,
            count_spin.valueChanged,
            lambda *_args: count_spin.value(),
            lambda value: transform_palette_state(
                "palette_diverging",
                lambda current: current.model_copy(update={"n_colors": int(value)}),
            ),
        )
        editor.add_form_row(
            layout,
            "Colors",
            editor.mixed_value_widget(count_spin, mixed=count_mixed, parent=page),
            "Number of colors in the generated diverging palette.",
        )

        center_mixed = editor.batch_is_mixed(
            operation, lambda target: target.palette_diverging.center
        )
        center_combo = editor.combo(
            ("light", "dark"),
            None if center_mixed else state.center,
            lambda center: transform_palette_state(
                "palette_diverging",
                lambda current: current.model_copy(update={"center": center}),
            ),
            parent=page,
            mixed=center_mixed,
            enabled=available,
        )
        center_combo.setObjectName("figureComposerSetPaletteDivergingCenterCombo")
        editor.add_form_row(
            layout,
            "Center",
            center_combo,
            "Use a light or dark color at the center of the palette.",
        )

    elif not mode_mixed and operation.palette_mode in {"light", "dark"}:
        mode = operation.palette_mode
        attribute = f"palette_{mode}"
        state = typing.cast(
            "FigureSequentialPaletteState", getattr(operation, attribute)
        )
        input_mixed = editor.batch_is_mixed(
            operation,
            operator.attrgetter(f"{attribute}.input"),
        )
        input_combo = QtWidgets.QComboBox(page)
        input_combo.setObjectName(f"figureComposerSetPalette{mode.title()}InputCombo")
        for input_space, label in _SEQUENTIAL_PALETTE_INPUT_LABELS.items():
            input_combo.addItem(label, input_space)
        if input_mixed:
            input_combo.insertItem(0, "(multiple values)", None)
            item = typing.cast("typing.Any", input_combo.model()).item(0)
            if item is not None:
                item.setEnabled(False)
            input_combo.setCurrentIndex(0)
        else:
            input_combo.setCurrentIndex(input_combo.findData(state.input))
        input_combo.setEnabled(available)

        def update_input(input_space: str) -> None:
            def convert(
                current: FigureSequentialPaletteState,
            ) -> FigureSequentialPaletteState:
                return current.model_copy(
                    update={
                        "input": input_space,
                        "color": _convert_palette_color(
                            typing.cast("typing.Any", sns),
                            current.color,
                            current.input,
                            input_space,
                        ),
                    }
                )

            transform_palette_state(attribute, convert, rebuild=True)

        editor.connect_signal(
            input_combo,
            input_combo.activated,
            lambda _index, combo=input_combo: (
                None
                if combo.currentData() is None
                else update_input(str(combo.currentData()))
            ),
        )
        editor.add_form_row(
            layout,
            "Input",
            input_combo,
            "Color space used to define the palette's seed color.",
        )

        color_mixed = input_mixed or editor.batch_is_mixed(
            operation,
            operator.attrgetter(f"{attribute}.color"),
        )
        seed_rgb = (
            _convert_palette_color(
                sns,
                state.color,
                state.input,
                "rgb",
            )
            if sns is not None
            else (0.0, 0.0, 0.0)
        )
        seed_color_button = _ColorPickerButton(
            page,
            color=_qt_color(seed_rgb),
            show_alpha_channel=False,
        )
        seed_color_button_ref = weakref.ref(seed_color_button)
        seed_color_button.setObjectName(
            f"figureComposerSetPalette{mode.title()}SeedColorButton"
        )
        seed_color_button.setAccessibleName("Choose seed color")
        seed_color_button.setToolTip("Choose the palette's seed color.")
        seed_color_button.setEnabled(available)

        def update_seed_color(color: QtGui.QColor) -> None:
            rgb = (color.redF(), color.greenF(), color.blueF())
            transform_palette_state(
                attribute,
                lambda current: current.model_copy(
                    update={
                        "color": _convert_palette_color(
                            sns,
                            rgb,
                            "rgb",
                            current.input,
                        )
                    }
                ),
                rebuild=True,
            )

        editor.connect_signal(
            seed_color_button,
            seed_color_button.colorSelected,
            update_seed_color,
        )
        editor.add_form_row(
            layout,
            "Seed color",
            editor.mixed_value_widget(
                seed_color_button,
                mixed=color_mixed,
                parent=page,
            ),
            "Choose the palette's seed color with the system color picker.",
        )

        if not input_mixed:
            if state.input == "husl":
                component_specs = (
                    ("Hue", 0.0, 359.0, 2, 1.0),
                    ("Saturation", 0.0, 99.0, 2, 1.0),
                    ("Lightness", 0.0, 99.0, 2, 1.0),
                )
            elif state.input == "hls":
                component_specs = (
                    ("Hue", 0.0, 1.0, 3, 0.01),
                    ("Lightness", 0.0, 1.0, 3, 0.01),
                    ("Saturation", 0.0, 1.0, 3, 0.01),
                )
            else:
                component_specs = (
                    ("Red", 0.0, 1.0, 3, 0.01),
                    ("Green", 0.0, 1.0, 3, 0.01),
                    ("Blue", 0.0, 1.0, 3, 0.01),
                )

            color_getter = operator.attrgetter(f"{attribute}.color")
            for index, (
                label,
                minimum,
                maximum,
                decimals,
                single_step,
            ) in enumerate(component_specs):
                mixed = editor.batch_is_mixed(
                    operation,
                    typing.cast(
                        "Callable[[FigureOperationState], float]",
                        lambda target, index=index: color_getter(target)[index],
                    ),
                )

                def update_component(
                    value: float, preview_only: bool, *, index: int = index
                ) -> None:
                    def replace_component(
                        current: FigureSequentialPaletteState,
                    ) -> FigureSequentialPaletteState:
                        color = list(current.color)
                        color[index] = float(value)
                        return current.model_copy(update={"color": tuple(color)})

                    transform_palette_state(
                        attribute, replace_component, preview_only=preview_only
                    )

                _add_palette_slider(
                    editor,
                    layout,
                    page,
                    label=label,
                    object_name=(
                        f"figureComposerSetPalette{mode.title()}{label}Control"
                    ),
                    value=state.color[index],
                    minimum=minimum,
                    maximum=maximum,
                    changed=update_component,
                    tooltip=f"{label} component of the {state.input.upper()} seed.",
                    mixed=mixed,
                    enabled=available,
                    decimals=decimals,
                    single_step=single_step,
                )

        count_mixed = editor.batch_is_mixed(
            operation,
            operator.attrgetter(f"{attribute}.n_colors"),
        )
        count_spin = QtWidgets.QSpinBox(page)
        count_spin.setObjectName(f"figureComposerSetPalette{mode.title()}NColorsSpin")
        count_spin.setAccessibleName("Colors")
        count_spin.setRange(3, 2_147_483_647)
        count_spin.setValue(state.n_colors)
        count_spin.setKeyboardTracking(False)
        count_spin.setEnabled(available)
        editor.connect_value_signal(
            count_spin,
            count_spin.valueChanged,
            lambda *_args: count_spin.value(),
            lambda value: transform_palette_state(
                attribute,
                lambda current: current.model_copy(update={"n_colors": int(value)}),
            ),
        )
        editor.add_form_row(
            layout,
            "Colors",
            editor.mixed_value_widget(count_spin, mixed=count_mixed, parent=page),
            f"Number of colors in the generated {mode} palette.",
        )

    refresh_preview()
    editor.add_form_row(
        layout,
        "Preview",
        preview,
        "Hover a color to see its hex code; right-click a color to copy it.",
    )
    return (
        StepSection(
            "palette",
            "Palette",
            page,
            "Choose the line color palette for later plotting steps.",
        ),
    )


def _section_summary(
    _tool: FigureComposerTool, key: str, operation: FigureOperationState
) -> str:
    if key == "palette":
        return _palette_display_text(operation)
    return ""


def _palette_call_code(operation: FigureOperationState) -> str | None:
    if not _palette_has_effect(operation):
        return None
    if operation.palette_mode == "cubehelix":
        state = operation.palette_cubehelix
        constructor_lines = (
            "    sns.cubehelix_palette(",
            f"        n_colors={state.n_colors!r},",
            f"        start={state.start!r},",
            f"        rot={state.rot!r},",
            f"        gamma={state.gamma!r},",
            f"        hue={state.hue!r},",
            f"        light={state.light!r},",
            f"        dark={state.dark!r},",
            f"        reverse={state.reverse!r},",
            "    )",
        )
        return "\n".join(("sns.set_palette(", *constructor_lines, ")"))
    if operation.palette_mode == "diverging":
        state = operation.palette_diverging
        constructor_lines = (
            "    sns.diverging_palette(",
            f"        h_neg={state.h_neg!r},",
            f"        h_pos={state.h_pos!r},",
            f"        s={state.s!r},",
            f"        l={state.l!r},",
            f"        sep={state.sep!r},",
            f"        n={state.n_colors!r},",
            f"        center={state.center!r},",
            "    )",
        )
        return "\n".join(("sns.set_palette(", *constructor_lines, ")"))
    if operation.palette_mode in {"light", "dark"}:
        state = (
            operation.palette_light
            if operation.palette_mode == "light"
            else operation.palette_dark
        )
        constructor_lines = (
            f"    sns.{operation.palette_mode}_palette(",
            f"        {state.color!r},",
            f"        n_colors={state.n_colors!r},",
            f"        input={state.input!r},",
            "    )",
        )
        return "\n".join(("sns.set_palette(", *constructor_lines, ")"))
    palette_value: str | list[str]
    if operation.palette_mode == "colors":
        palette_value = list(operation.palette_colors)
    else:
        palette_value = operation.palette_name
    args = [repr(palette_value)]
    for key, value in _palette_call_kwargs(operation).items():
        args.append(f"{key}={value!r}")
    return f"sns.set_palette({', '.join(args)})"


def _code_lines(
    _tool: FigureComposerTool, operation: FigureOperationState
) -> list[str]:
    palette_call = _palette_call_code(operation)
    if palette_call is None:
        return []
    return palette_call.splitlines()


def _required_imports(
    _tool: FigureComposerTool, operation: FigureOperationState
) -> tuple[str, ...]:
    if not _palette_has_effect(operation):
        return ()
    return ("import seaborn as sns",)


SPEC = OperationSpec(
    kind=FigureOperationKind.SET_PALETTE,
    add_actions=(
        AddStepActionSpec(
            action_id=FigureOperationKind.SET_PALETTE.value,
            text="Set Palette",
            tooltip="Set the line color cycle with seaborn.set_palette.",
            create_operation=_create_operation,
        ),
    ),
    display_text=_display_text,
    tooltip=_tooltip,
    target_text=_target_text,
    has_invalid_target=_no_invalid_target,
    uses_source_section=_uses_no_source_section,
    build_source_editor=_empty_source_editor,
    build_editor_sections=_editor_sections,
    section_summary=_section_summary,
    render=_render_set_palette,
    code_lines=_code_lines,
    render_cache_safe=_always_render_cache_safe,
    required_imports=_required_imports,
)
