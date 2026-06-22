"""Seaborn palette operation for Figure Composer."""

from __future__ import annotations

import typing

from qtpy import QtGui, QtWidgets

from erlab.interactive._figurecomposer._operations._base import (
    AddStepActionSpec,
    OperationSpec,
    StepSection,
    _empty_source_editor,
    _empty_source_names,
    _no_invalid_target,
    _uses_no_axes,
    _uses_no_source_section,
)
from erlab.interactive._figurecomposer._state import (
    FigureOperationKind,
    FigureOperationState,
)

if typing.TYPE_CHECKING:
    from matplotlib.figure import Figure

    from erlab.interactive._figurecomposer._tool import FigureComposerTool

_PALETTE_OPTIONS = (
    "deep",
    "muted",
    "pastel",
    "bright",
    "dark",
    "colorblind",
    "tab10",
    "tab20",
    "Set1",
    "Set2",
    "Set3",
    "Paired",
    "rocket",
    "mako",
    "flare",
    "crest",
    "viridis",
    "plasma",
    "magma",
    "inferno",
    "cividis",
    "icefire",
    "vlag",
)
_DESAT_AUTO_VALUE = -0.01
_PALETTE_COUNT_MAX = 256


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
        layout = self.layout()
        if layout is None:  # pragma: no cover
            return
        while layout.count():
            item = layout.takeAt(0)
            if item is not None and (widget := item.widget()) is not None:
                widget.setParent(None)
                widget.deleteLater()
        if not colors:
            label = QtWidgets.QLabel("No preview", self)
            label.setObjectName("figureComposerSetPalettePreviewUnavailable")
            label.setEnabled(False)
            layout.addWidget(label)
            layout.addStretch(1)
            return
        for index, color in enumerate(colors):
            swatch = QtWidgets.QFrame(self)
            swatch.setObjectName("figureComposerSetPalettePreviewSwatch")
            swatch.setProperty("palette_color_index", index)
            qcolor = _qt_color(color)
            swatch.setProperty("palette_color", qcolor.name())
            swatch.setAccessibleName(f"Palette color {index + 1}")
            swatch.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
            swatch.setMinimumSize(18, 18)
            swatch.setMaximumSize(28, 18)
            swatch.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Fixed,
            )
            border = self.palette().color(QtGui.QPalette.ColorRole.Mid).name()
            swatch.setStyleSheet(
                f"background-color: {qcolor.name()}; border: 1px solid {border};"
            )
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


def _palette_options(operation: FigureOperationState) -> tuple[str, ...]:
    if operation.palette_name in _PALETTE_OPTIONS:
        return _PALETTE_OPTIONS
    return (*_PALETTE_OPTIONS, operation.palette_name)


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
    palette_name: str,
    n_colors: int | None,
    desat: float | None,
) -> tuple[typing.Any, ...]:
    if sns is None:
        return ()
    try:
        return tuple(sns.color_palette(palette_name, n_colors=n_colors, desat=desat))
    except (TypeError, ValueError):
        return ()


def _apply_palette_to_existing_axes(fig: Figure, sns: typing.Any) -> None:
    colors = sns.color_palette()
    for axis in fig.axes:
        axis.set_prop_cycle(color=colors)


def _render_set_palette(
    _tool: FigureComposerTool,
    operation: FigureOperationState,
    fig: Figure,
    _axs: typing.Any,
) -> None:
    sns = _import_seaborn()
    if sns is None:
        return
    sns.set_palette(operation.palette_name, **_palette_call_kwargs(operation))
    _apply_palette_to_existing_axes(fig, sns)


def _create_operation(_tool: FigureComposerTool) -> FigureOperationState:
    return FigureOperationState.set_palette(label="set palette")


def _display_text(_tool: FigureComposerTool, operation: FigureOperationState) -> str:
    if _import_seaborn() is None:
        return f"Skipped Set Palette: {operation.palette_name}"
    return f"Set Palette: {operation.palette_name}"


def _tooltip(_tool: FigureComposerTool, operation: FigureOperationState) -> str:
    if _import_seaborn() is None:
        return "Install seaborn to apply this palette step.\nTargets: figure"
    return (
        "Sets the Matplotlib line color cycle with seaborn.set_palette().\n"
        f"Palette: {operation.palette_name}\n"
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
        operation.palette_name,
        operation.palette_n_colors,
        operation.palette_desat,
    ]
    unchanged = object()

    def refresh_preview(
        *,
        palette_name: str | None = None,
        n_colors: typing.Any = unchanged,
        desat: typing.Any = unchanged,
    ) -> None:
        if palette_name is not None:
            preview_values[0] = palette_name
        if n_colors is not unchanged:
            preview_values[1] = n_colors
        if desat is not unchanged:
            preview_values[2] = desat
        preview.set_colors(
            _palette_colors(
                sns,
                preview_values[0],
                preview_values[1],
                preview_values[2],
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

    palette_mixed = tool._batch_is_mixed(operation, lambda target: target.palette_name)
    palette_combo = tool._combo(
        _palette_options(operation),
        None if palette_mixed else operation.palette_name,
        lambda text: (
            refresh_preview(palette_name=text),
            tool._update_current_operation(palette_name=text),
        ),
        parent=page,
        mixed=palette_mixed,
        enabled=available,
    )
    palette_combo.setObjectName("figureComposerSetPaletteNameCombo")
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
    tool._connect_value_signal(
        count_spin,
        count_spin.valueChanged,
        lambda *_args: count_spin.value(),
        lambda value: (
            refresh_preview(n_colors=None if value == 0 else int(value)),
            tool._update_current_operation(
                palette_n_colors=None if value == 0 else int(value)
            ),
        ),
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
    tool._connect_value_signal(
        desat_spin,
        desat_spin.valueChanged,
        lambda *_args: desat_spin.value(),
        lambda value: (
            refresh_preview(desat=None if value <= _DESAT_AUTO_VALUE else float(value)),
            tool._update_current_operation(
                palette_desat=None if value <= _DESAT_AUTO_VALUE else float(value)
            ),
        ),
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
        "Colors that seaborn resolves for the current palette settings.",
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
        return operation.palette_name
    return ""


def _palette_call_code(operation: FigureOperationState) -> str:
    args = [repr(operation.palette_name)]
    for key, value in _palette_call_kwargs(operation).items():
        args.append(f"{key}={value!r}")
    return f"sns.set_palette({', '.join(args)})"


def _code_lines(
    _tool: FigureComposerTool, operation: FigureOperationState
) -> list[str]:
    return [
        "try:",
        "    import seaborn as sns",
        "except ImportError:",
        "    pass",
        "else:",
        f"    {_palette_call_code(operation)}",
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
    source_names=_empty_source_names,
    build_source_editor=_empty_source_editor,
    build_editor_sections=_editor_sections,
    section_summary=_section_summary,
    render=_render_set_palette,
    code_lines=_code_lines,
    required_imports=_required_imports,
)
