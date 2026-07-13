"""Seaborn palette operation for Figure Composer."""

from __future__ import annotations

import functools
import html
import typing

import matplotlib.pyplot as plt
from qtpy import QtCore, QtGui, QtWidgets

from erlab.interactive._figurecomposer._operations._base import (
    AddStepActionSpec,
    OperationSpec,
    StepSection,
    _empty_source_editor,
    _no_invalid_target,
    _uses_no_axes,
    _uses_no_source_section,
)
from erlab.interactive._figurecomposer._state import (
    FigureOperationKind,
    FigureOperationState,
)
from erlab.interactive._figurecomposer._widgets import _ColorListEditorWidget
from erlab.plotting.colors import close_to_white

if typing.TYPE_CHECKING:
    from matplotlib.figure import Figure

    from erlab.interactive._figurecomposer._tool import FigureComposerTool

_SET_PALETTE_DOC_URL = "https://seaborn.pydata.org/generated/seaborn.set_palette.html"
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
_PALETTE_ICON_SIZE = QtCore.QSize(72, 14)
_PALETTE_ICON_COLORS = 8
_PALETTE_MODE_LABELS = {
    "named": "Named palette",
    "colors": "Custom colors",
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
        clipboard = QtWidgets.QApplication.clipboard()
        if clipboard is not None:  # pragma: no branch
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

    def set_colors(self, colors: typing.Sequence[typing.Any]) -> None:
        base_layout = self.layout()
        if base_layout is None:  # pragma: no cover
            return
        layout = typing.cast("QtWidgets.QHBoxLayout", base_layout)
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


def _palette_hex_colors(colors: typing.Sequence[typing.Any]) -> tuple[str, ...]:
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
    width: int = _PALETTE_ICON_SIZE.width(),
    height: int = _PALETTE_ICON_SIZE.height(),
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
    combo.setIconSize(_PALETTE_ICON_SIZE)
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


def _editor_sections(
    tool: FigureComposerTool, operation: FigureOperationState
) -> tuple[StepSection, ...]:
    page, layout = tool._new_step_form_page("figureComposerSetPalettePage")
    sns = _import_seaborn()
    available = sns is not None
    tool._add_form_section(
        layout,
        "Palette",
        object_name="figureComposerSetPaletteSection",
    )

    preview = _PalettePreviewWidget(page)
    preview.setEnabled(available)
    preview_values: list[typing.Any] = [
        operation.palette_mode,
        operation.palette_name,
        operation.palette_colors,
        operation.palette_n_colors,
        operation.palette_desat,
    ]
    unchanged = object()

    def refresh_preview(
        *,
        palette_mode: str | None = None,
        palette_name: str | None = None,
        palette_colors: typing.Any = unchanged,
        n_colors: typing.Any = unchanged,
        desat: typing.Any = unchanged,
    ) -> None:
        if palette_mode is not None:
            preview_values[0] = palette_mode
        if palette_name is not None:
            preview_values[1] = palette_name
        if palette_colors is not unchanged:
            preview_values[2] = tuple(palette_colors)
        if n_colors is not unchanged:
            preview_values[3] = n_colors
        if desat is not unchanged:
            preview_values[4] = desat
        palette_value = (
            preview_values[2] if preview_values[0] == "colors" else preview_values[1]
        )
        preview.set_colors(
            _palette_colors(
                sns,
                palette_value,
                preview_values[3],
                preview_values[4],
            )
        )

    if not available:
        message = QtWidgets.QLabel(
            "Install seaborn to preview or apply palettes.", page
        )
        message.setObjectName("figureComposerSetPaletteUnavailableLabel")
        message.setProperty("missing_dependency", "seaborn")
        message.setWordWrap(True)
        tool._add_form_row(
            layout,
            "Dependency",
            message,
            "This step needs seaborn in the active Python environment.",
        )

    def update_palette_mode(mode: str) -> None:
        if mode not in _PALETTE_MODE_LABELS:
            return
        updates: dict[str, typing.Any] = {"palette_mode": mode}
        if mode == "colors" and not operation.palette_colors:
            seeded_colors = _palette_hex_colors(
                _palette_colors(
                    sns,
                    operation.palette_name,
                    operation.palette_n_colors,
                    operation.palette_desat,
                )
            )
            if seeded_colors:
                updates["palette_colors"] = seeded_colors
        refresh_preview(
            palette_mode=mode,
            palette_colors=updates.get("palette_colors", operation.palette_colors),
        )
        tool._update_current_operation_rebuild(**updates)

    mode_mixed = tool._batch_is_mixed(operation, lambda target: target.palette_mode)
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
        "Choose a named seaborn or Matplotlib palette, or provide explicit colors."
    )
    tool._connect_editor_signal(
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
    docs_button.setProperty("figure_palette_doc_url", _SET_PALETTE_DOC_URL)
    docs_button.setToolTip("Open seaborn.set_palette documentation.")

    def open_docs(_checked: bool = False) -> None:
        QtGui.QDesktopServices.openUrl(QtCore.QUrl(_SET_PALETTE_DOC_URL))

    tool._connect_editor_signal(
        docs_button,
        docs_button.clicked,
        open_docs,
    )
    mode_layout.addWidget(docs_button)
    tool._add_form_row(
        layout,
        "Use",
        mode_row,
        "Choose whether this step uses a named palette or explicit colors.",
    )

    def update_palette_name(text: str) -> None:
        refresh_preview(palette_name=text)
        tool._update_current_operation(palette_name=text)

    if not mode_mixed and operation.palette_mode == "colors":

        def update_palette_colors(colors: typing.Any) -> None:
            colors = tuple(str(color).strip() for color in colors if str(color).strip())
            refresh_preview(palette_colors=colors)
            tool._update_current_operation(
                palette_mode="colors",
                palette_colors=colors,
            )

        colors_mixed = tool._batch_is_mixed(
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
        tool._connect_value_signal(
            colors_widget,
            colors_widget.colorsChanged,
            lambda colors: tuple(colors),
            update_palette_colors,
            unchanged_mixed=colors_widget.batchUnchanged,
        )
        tool._add_form_row(
            layout,
            "Colors",
            colors_widget,
            "Explicit color sequence passed to seaborn.set_palette.",
        )
    else:
        palette_mixed = tool._batch_is_mixed(
            operation, lambda target: target.palette_name
        )
        palette_combo = tool._combo(
            _palette_options(operation, sns),
            None if palette_mixed else operation.palette_name,
            update_palette_name,
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
        tool._add_form_row(
            layout,
            "Palette",
            palette_combo,
            "Named seaborn or Matplotlib palette passed to seaborn.set_palette.",
        )

    count_mixed = tool._batch_is_mixed(
        operation, lambda target: target.palette_n_colors
    )
    count_spin = QtWidgets.QSpinBox(page)
    count_spin.setObjectName("figureComposerSetPaletteCountSpin")
    count_spin.setRange(0, _PALETTE_COUNT_MAX)
    count_spin.setSpecialValueText("Auto")
    count_spin.setValue(operation.palette_n_colors or 0)
    count_spin.setEnabled(available)

    def update_count(value: typing.Any) -> None:
        n_colors = None if value == 0 else int(value)
        refresh_preview(n_colors=n_colors)
        tool._update_current_operation(palette_n_colors=n_colors)

    tool._connect_value_signal(
        count_spin,
        count_spin.valueChanged,
        lambda *_args: count_spin.value(),
        update_count,
    )

    desat_mixed = tool._batch_is_mixed(operation, lambda target: target.palette_desat)
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
        desat = None if value <= _DESAT_AUTO_VALUE else float(value)
        refresh_preview(desat=desat)
        tool._update_current_operation(palette_desat=desat)

    tool._connect_value_signal(
        desat_spin,
        desat_spin.valueChanged,
        lambda *_args: desat_spin.value(),
        update_desat,
    )

    tool._add_compound_form_row(
        layout,
        "Options",
        (
            (
                "Colors",
                tool._mixed_value_widget(count_spin, mixed=count_mixed, parent=page),
                "Number of colors requested from the palette. Auto uses seaborn's "
                "default length.",
            ),
            (
                "Saturation",
                tool._mixed_value_widget(desat_spin, mixed=desat_mixed, parent=page),
                "Scale palette saturation. Auto uses seaborn's default saturation.",
            ),
        ),
        "Optional arguments passed to seaborn.set_palette.",
    )

    color_codes_mixed = tool._batch_is_mixed(
        operation, lambda target: target.palette_color_codes
    )
    color_codes_check = tool._check_box(
        operation.palette_color_codes,
        lambda checked: tool._update_current_operation(palette_color_codes=checked),
        parent=page,
        mixed=color_codes_mixed,
    )
    color_codes_check.setObjectName("figureComposerSetPaletteColorCodesCheck")
    color_codes_check.setEnabled(available)
    tool._add_form_row(
        layout,
        "Color codes",
        color_codes_check,
        "Map matplotlib color-code letters such as b, g, and r to this palette.",
    )

    refresh_preview()
    tool._add_form_row(
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
    return [
        "try:",
        "    import seaborn as sns",
        "except ImportError:",
        "    pass",
        "else:",
        f"    {palette_call}",
        "    for ax in fig.axes:",
        "        ax.set_prop_cycle(color=sns.color_palette())",
    ]


def _required_imports(
    _tool: FigureComposerTool, _operation: FigureOperationState
) -> tuple[str, ...]:
    return ()


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
    uses_axes=_uses_no_axes,
    uses_source_section=_uses_no_source_section,
    build_source_editor=_empty_source_editor,
    build_editor_sections=_editor_sections,
    section_summary=_section_summary,
    render=_render_set_palette,
    code_lines=_code_lines,
    required_imports=_required_imports,
)
