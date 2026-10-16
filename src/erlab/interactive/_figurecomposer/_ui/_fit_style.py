"""Fit-plot style controls for Figure Composer."""

from __future__ import annotations

import ast
import math
import typing

from qtpy import QtCore, QtGui, QtWidgets

from erlab.interactive._figurecomposer._exceptions import FigureComposerInputError
from erlab.interactive._figurecomposer._line_style import (
    LINE_MARKER_OPTIONS,
    LINE_STYLE_OPTIONS,
    color_kw_value_from_text,
    normalize_style_value,
)
from erlab.interactive._figurecomposer._text import _dict_from_text, _format_dict
from erlab.interactive._figurecomposer._ui._color_widgets import _ColorLineEditWidget

if typing.TYPE_CHECKING:
    from collections.abc import Callable

    from erlab.interactive._figurecomposer._model._state import FigureOperationState
    from erlab.interactive._figurecomposer._ui._operation_editor import (
        FigureOperationEditor,
    )


_UNSET = object()
_ALIASES = {
    "color": ("c",),
    "linestyle": ("ls",),
    "linewidth": ("lw",),
    "markersize": ("ms",),
    "edgecolor": ("ec",),
    "facecolor": ("fc", "color"),
}
_ERROR_KEYS = frozenset(
    {"ecolor", "elinewidth", "capsize", "capthick", "barsabove", "errorevery"}
)
_RESERVED_KEYS = frozenset({"x", "y", "xerr", "yerr", "fmt", "data"})


def _style(
    operation: FigureOperationState, field: str, component: str | None
) -> dict[str, typing.Any]:
    styles = getattr(operation, field)
    return styles if component is None else styles.get(component, {})


def _style_value(
    operation: FigureOperationState, field: str, component: str | None, key: str
) -> typing.Any:
    styles = _style(operation, field, component)
    for candidate in (key, *_ALIASES.get(key, ())):
        if candidate in styles:
            return styles[candidate]
    return None


def _update_style(
    editor: FigureOperationEditor,
    field: str,
    component: str | None,
    key: str,
    value: typing.Any,
) -> None:
    def update(_index: int, operation: FigureOperationState) -> FigureOperationState:
        styles = dict(_style(operation, field, component))
        for candidate in (key, *_ALIASES.get(key, ())):
            styles.pop(candidate, None)
        if value is not _UNSET:
            styles[key] = value
        if component is None:
            return operation.model_copy(update={field: styles})
        components = dict(getattr(operation, field))
        if styles:
            components[component] = styles
        else:
            components.pop(component, None)
        return operation.model_copy(update={field: components})

    editor.request_transform(update)


def _number(
    text: str, *, minimum: float | None = None, maximum: float | None = None
) -> typing.Any:
    text = text.strip()
    if not text:
        return _UNSET
    try:
        value = float(text)
    except ValueError as exc:
        raise FigureComposerInputError(
            "Enter a number or leave empty for default."
        ) from exc
    if not math.isfinite(value):
        raise FigureComposerInputError("Enter a finite number.")
    if minimum is not None and value < minimum:
        raise FigureComposerInputError(f"Enter a value of {minimum:g} or more.")
    if maximum is not None and value > maximum:
        raise FigureComposerInputError(f"Enter a value of {maximum:g} or less.")
    return value


def _property_value(text: str) -> typing.Any:
    text = text.strip()
    if not text:
        return _UNSET
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError) as exc:
        if text[0] in "[({\"'" or text in {"True", "False", "None"}:
            raise FigureComposerInputError(
                "Enter a valid value, such as red, 2, or (0, 2)."
            ) from exc
        return text


def _error_every(text: str) -> typing.Any:
    value = _property_value(text)
    if value is _UNSET:
        return value
    if type(value) is int and value > 0:
        return value
    if (
        isinstance(value, list | tuple)
        and len(value) == 2
        and all(type(item) is int for item in value)
        and value[0] >= 0
        and value[1] > 0
    ):
        return tuple(value)
    raise FigureComposerInputError("Enter a positive stride or a (start, stride) pair.")


def _text_control(
    editor: FigureOperationEditor,
    operation: FigureOperationState,
    page: QtWidgets.QWidget,
    layout: QtWidgets.QFormLayout,
    field: str,
    component: str | None,
    key: str,
    title: str,
    object_name: str,
    parser: Callable[[str], typing.Any],
    tooltip: str,
    *,
    color: bool = False,
) -> None:
    text, mixed = editor.batch_text(
        operation,
        lambda target: _style_value(target, field, component, key),
        lambda value: "" if value is None else str(value),
    )
    if color:
        widget: QtWidgets.QWidget = _ColorLineEditWidget(text, page)
        color_widget = typing.cast("_ColorLineEditWidget", widget)
        edit = color_widget.line_edit
        color_widget.setLineEditObjectName(object_name)
        color_widget.setColorButtonObjectName(f"{object_name}Button")
        signal = color_widget.editingFinished
        getter = color_widget.text
    else:
        widget = edit = editor.line_edit(text, parent=page)
        edit.setObjectName(object_name)
        signal = edit.editingFinished
        getter = edit.text
    if not mixed:
        edit.setPlaceholderText("default")
    editor.apply_mixed_line_edit(edit, mixed)
    editor.connect_value_signal(
        widget,
        signal,
        getter,
        lambda text: _update_style(editor, field, component, key, parser(text)),
        unchanged_mixed=lambda: editor.line_edit_batch_unchanged(edit),
    )
    editor.add_form_row(layout, title, widget, tooltip)


def _additional_properties(
    editor: FigureOperationEditor,
    operation: FigureOperationState,
    page: QtWidgets.QWidget,
    layout: QtWidgets.QFormLayout,
    field: str,
    component: str | None,
    prefix: str,
    controlled: set[str],
    *,
    line: bool,
    errorbars: bool,
) -> None:
    preserved = controlled.copy()
    if field == "fit_data_kw":
        preserved |= _ERROR_KEYS
    blocked = preserved | _RESERVED_KEYS
    text, mixed = editor.batch_text(
        operation,
        lambda target: {
            key: value
            for key, value in _style(target, field, component).items()
            if key not in blocked
        },
        _format_dict,
    )
    row = QtWidgets.QWidget(page)
    row_layout = QtWidgets.QHBoxLayout(row)
    row_layout.setContentsMargins(0, 0, 0, 0)
    kwargs_edit = editor.line_edit(text, parent=row)
    kwargs_edit.setObjectName(f"{prefix}KwargsEdit")
    kwargs_edit.setPlaceholderText("optional")
    editor.apply_mixed_line_edit(kwargs_edit, mixed)

    def update_kwargs(text: str) -> None:
        kwargs: dict[typing.Any, typing.Any] = _dict_from_text(text)
        if any(not isinstance(key, str) or not key.isidentifier() for key in kwargs):
            raise FigureComposerInputError("Enter Matplotlib keyword argument names.")
        invalid = sorted(kwargs.keys() & blocked)
        if invalid:
            raise FigureComposerInputError(
                "These kwargs have dedicated controls or are supplied by the fit: "
                + ", ".join(invalid)
                + "."
            )

        def update(_index: int, target: FigureOperationState) -> FigureOperationState:
            styles = {
                key: value
                for key, value in _style(target, field, component).items()
                if key in preserved
            } | kwargs
            if component is None:
                return target.model_copy(update={field: styles})
            components = dict(getattr(target, field))
            if styles:
                components[component] = styles
            else:
                components.pop(component, None)
            return target.model_copy(update={field: components})

        editor.request_transform(update)

    editor.connect_line_edit_finished(kwargs_edit, update_kwargs)
    row_layout.addWidget(kwargs_edit, 1)
    method = "errorbar" if errorbars else "plot" if line else "fill_between"
    docs_url = (
        f"https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.{method}.html"
    )
    docs_button = QtWidgets.QToolButton(row)
    docs_button.setObjectName(f"{prefix}KwargsDocsButton")
    docs_button.setText("Docs")
    docs_button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextOnly)
    docs_button.setProperty("figure_fit_style_doc_url", docs_url)
    docs_button.setToolTip(f"Open Matplotlib Axes.{method} documentation.")

    def open_docs(_checked: bool = False) -> None:
        QtGui.QDesktopServices.openUrl(QtCore.QUrl(docs_url))

    editor.connect_signal(
        docs_button,
        docs_button.clicked,
        open_docs,
    )
    row_layout.addWidget(docs_button)
    tooltip = (
        "Additional Matplotlib kwargs not covered by the controls above.\n"
        "Enter keyword arguments or a dictionary literal. Leave empty to clear."
    )
    kwargs_edit.setToolTip(tooltip)
    editor.add_form_row(layout, "Line kwargs" if line else "Extra kwargs", row, tooltip)


def add_fit_style_controls(
    editor: FigureOperationEditor,
    operation: FigureOperationState,
    page: QtWidgets.QWidget,
    layout: QtWidgets.QFormLayout,
    field: str,
    *,
    prefix: str,
    line: bool,
    component: str | None = None,
    errorbars: bool = False,
) -> None:
    """Add independent controls while preserving unedited style properties."""
    if component is not None:
        prefix = f"{prefix}:{component}:"
    controlled = {"label", "color", "alpha", "zorder", "linewidth", "linestyle"}
    if line:
        controlled |= {"marker", "markersize"}
    else:
        controlled |= {"edgecolor", "facecolor"}
    controlled |= {
        alias for key in tuple(controlled) for alias in _ALIASES.get(key, ())
    }
    _text_control(
        editor,
        operation,
        page,
        layout,
        field,
        component,
        "label",
        "Label",
        f"{prefix}Label",
        lambda text: text or _UNSET,
        "Label for this part of the fit plot. Leave empty to use its default label.",
    )
    _text_control(
        editor,
        operation,
        page,
        layout,
        field,
        component,
        "color" if line else "facecolor",
        "Color" if line else "Fill color",
        f"{prefix}Color",
        lambda text: color_kw_value_from_text(text) if text.strip() else _UNSET,
        "Leave empty to inherit the default color.",
        color=True,
    )
    if not line:
        _text_control(
            editor,
            operation,
            page,
            layout,
            field,
            component,
            "edgecolor",
            "Edge color",
            f"{prefix}Edgecolor",
            lambda text: color_kw_value_from_text(text) if text.strip() else _UNSET,
            "Outline color of the band.",
            color=True,
        )
    for key, title, options in (
        *((("marker", "Marker", LINE_MARKER_OPTIONS),) if line else ()),
        ("linestyle", "Line style", LINE_STYLE_OPTIONS),
    ):

        def style_value(target: FigureOperationState, key: str = key) -> str | None:
            return normalize_style_value(_style_value(target, field, component, key))

        def style_changed(text: str, key: str = key) -> None:
            _update_style(
                editor, field, component, key, _UNSET if text == "default" else text
            )

        current = editor.batch_combo_text(
            operation,
            style_value,
            lambda item: "default" if item is None else str(item),
        )
        values = ("default", *options)
        if current is not None and current not in values:
            values = (*values, current)
        combo = editor.combo(
            values,
            current,
            style_changed,
            parent=page,
            mixed=current is None,
        )
        combo.setObjectName(f"{prefix}{key.capitalize()}")
        editor.add_form_row(
            layout,
            title,
            combo,
            f"Matplotlib {title.lower()}. Choose default to inherit.",
        )
    for key, title, minimum, maximum in (
        ("linewidth", "Line width", 0.0, None),
        *((("markersize", "Marker size", 0.0, None),) if line else ()),
        ("alpha", "Alpha", 0.0, 1.0),
        ("zorder", "Z order", None, None),
    ):

        def parse_number(
            text: str, minimum: float | None = minimum, maximum: float | None = maximum
        ) -> typing.Any:
            return _number(text, minimum=minimum, maximum=maximum)

        _text_control(
            editor,
            operation,
            page,
            layout,
            field,
            component,
            key,
            title,
            f"{prefix}{key.capitalize()}",
            parse_number,
            "Leave empty to inherit. "
            + (
                "Higher values draw on top."
                if key == "zorder"
                else "Sizes are in points."
                if key in {"linewidth", "markersize"}
                else "Opacity ranges from 0 to 1."
            ),
        )
    _additional_properties(
        editor,
        operation,
        page,
        layout,
        field,
        component,
        prefix,
        controlled,
        line=line,
        errorbars=errorbars,
    )


def add_fit_error_controls(
    editor: FigureOperationEditor,
    operation: FigureOperationState,
    page: QtWidgets.QWidget,
    layout: QtWidgets.QFormLayout,
) -> None:
    """Add error-bar styles only for sources that contain fit weights."""
    prefix = "figureComposerFitPlotError"
    field = "fit_data_kw"
    _text_control(
        editor,
        operation,
        page,
        layout,
        field,
        None,
        "ecolor",
        "Error color",
        f"{prefix}Color",
        lambda text: color_kw_value_from_text(text) if text.strip() else _UNSET,
        "Leave empty to use the data color.",
        color=True,
    )
    for key, title, suffix in (
        ("elinewidth", "Error line width", "Linewidth"),
        ("capsize", "Cap size", "Capsize"),
        ("capthick", "Cap thickness", "Capthick"),
    ):
        _text_control(
            editor,
            operation,
            page,
            layout,
            field,
            None,
            key,
            title,
            f"{prefix}{suffix}",
            lambda text: _number(text, minimum=0.0),
            "Size in points. Leave empty to use the default."
            + (
                " Marker edge width overrides cap thickness."
                if key == "capthick"
                else ""
            ),
        )
    current = editor.batch_combo_text(
        operation,
        lambda target: target.fit_data_kw.get("barsabove"),
        lambda item: "default" if item is None else "above" if item else "below",
    )
    barsabove = editor.labeled_combo(
        ("default", "above", "below"),
        ("Default", "Above data", "Below data"),
        current,
        lambda text: _update_style(
            editor,
            field,
            None,
            "barsabove",
            _UNSET if text == "default" else text == "above",
        ),
        parent=page,
        mixed=current is None,
    )
    barsabove.setObjectName(f"{prefix}Barsabove")
    editor.add_form_row(
        layout,
        "Bars above data",
        barsabove,
        "Draw the error bars above the data markers.",
    )
    _text_control(
        editor,
        operation,
        page,
        layout,
        field,
        None,
        "errorevery",
        "Error bar stride",
        f"{prefix}Errorevery",
        _error_every,
        "Show every nth error bar, or enter (start, stride). "
        "Leave empty to show all error bars.",
    )
