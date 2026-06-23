"""Legend label helpers for line-like Figure Composer operations."""

from __future__ import annotations

import json
import re
import typing

import numpy as np

import erlab.plotting.annotations as plot_annotations
from erlab.interactive._figurecomposer._state import (
    FigureMethodFamily,
    FigureOperationKind,
    FigureOperationState,
)
from erlab.interactive._figurecomposer._text import (
    _format_string_tuple,
    _string_tuple_from_text,
)

if typing.TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import xarray as xr

    from erlab.interactive._figurecomposer._tool import FigureComposerTool


_PLACEHOLDER_RE = re.compile(
    r"\{(?P<name>[^{}!:]+?)(?:!(?P<conversion>[rsa]))?"
    r"(?::(?P<format>[^{}]*))?\}"
)
_GENERIC_PLACEHOLDER_ORDER = ("value", "dim", "number", "index", "source")
_LABEL_FIELD_SOURCES_KEY = "__figure_composer_label_field_sources__"
_LABEL_INTERNAL_KEYS = frozenset({_LABEL_FIELD_SOURCES_KEY})


class _LabelPlaceholderError(ValueError):
    """Raised when a label placeholder is explicit but unavailable."""


def _field_names(text: str) -> set[str]:
    return {match.group("name") for match in _PLACEHOLDER_RE.finditer(text)}


def _explicit_field_names(text: str) -> set[str]:
    return {
        match.group("name")
        for match in _PLACEHOLDER_RE.finditer(text)
        if _placeholder_is_explicit(match)
    }


def _placeholder_is_explicit(match: re.Match[str]) -> bool:
    return bool(match.group("conversion") or match.group("format") is not None)


def _available_field_names(
    contexts: Sequence[Mapping[str, typing.Any]],
) -> set[str]:
    return set().union(
        *(
            {name for name in context if name not in _LABEL_INTERNAL_KEYS}
            for context in contexts
        )
    )


def _scalar_value(value: typing.Any) -> typing.Any | None:
    array = np.asarray(value)
    if array.size != 1:
        return None
    try:
        return array.reshape(()).item()
    except ValueError:
        return None


def label_context(
    profile: xr.DataArray | None = None,
    *,
    index: int,
    source: str | None = None,
    dim: str | None = None,
    value: typing.Any = None,
) -> dict[str, typing.Any]:
    context: dict[str, typing.Any] = {
        "index": index,
        "number": index + 1,
    }
    if source is not None:
        context["source"] = source
    if dim is not None:
        context["dim"] = dim
    if value is not None:
        scalar = _scalar_value(value)
        context["value"] = value if scalar is None else scalar
    if profile is not None:
        field_sources: dict[str, str] = {}
        for name, coord in profile.coords.items():
            scalar = _scalar_value(coord.values)
            if scalar is not None:
                field = str(name)
                context[field] = scalar
                field_sources[field] = "coord"
        for name, value in profile.attrs.items():
            field = str(name)
            if field in context:
                continue
            scalar = _scalar_value(value)
            if scalar is not None:
                context[field] = scalar
                field_sources[field] = "attr"
        if field_sources:
            context[_LABEL_FIELD_SOURCES_KEY] = field_sources
    return context


def labels_from_text(
    text: str,
    contexts: Sequence[Mapping[str, typing.Any]],
    *,
    literal_values: Sequence[str] = (),
    default: str | None = None,
    item_name: str = "profile",
) -> tuple[str | None, ...]:
    count = len(contexts)
    if count < 1:
        return ()
    stripped = text.strip()
    if not stripped:
        return _literal_label_values(literal_values, count, default=default)

    fields = _field_names(text)
    available_fields = _available_field_names(contexts)
    recognized_fields = fields & available_fields
    if recognized_fields or _explicit_field_names(text):
        labels: list[str] = []
        for context in contexts:
            try:
                labels.append(
                    _format_label_text(
                        text,
                        dict(context),
                        available_fields=available_fields,
                        item_name=item_name,
                    )
                )
            except _LabelPlaceholderError:
                raise
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Could not format legend label {text!r} for this {item_name}"
                ) from exc
        return tuple(labels)

    return _literal_label_values(_string_tuple_from_text(text), count, default=default)


def label_text_uses_placeholders(
    text: str, contexts: Sequence[Mapping[str, typing.Any]]
) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    fields = _field_names(text)
    if not fields:
        return False
    available_fields = _available_field_names(contexts)
    return bool((fields & available_fields) or _explicit_field_names(text))


def label_field_names(text: str) -> set[str]:
    return _field_names(text)


def label_fstring_code(text: str, field_expressions: Mapping[str, str]) -> str:
    parts = ["f'"]
    previous_end = 0
    for match in _PLACEHOLDER_RE.finditer(text):
        parts.append(_fstring_literal(text[previous_end : match.start()]))
        name = match.group("name")
        if name not in field_expressions:
            if _placeholder_is_explicit(match):
                raise _unavailable_placeholder_error(
                    match,
                    set(field_expressions),
                    item_name="generated code",
                )
            parts.append(_fstring_literal(match.group(0)))
            previous_end = match.end()
            continue
        field_code = "{" + field_expressions[name]
        if conversion := match.group("conversion"):
            field_code += f"!{conversion}"
        if format_spec := match.group("format"):
            field_code += f":{format_spec}"
        field_code += "}"
        parts.append(field_code)
        previous_end = match.end()
    parts.append(_fstring_literal(text[previous_end:]))
    parts.append("'")
    return "".join(parts)


def coord_value_expression(coord_name: str, *, profile_name: str = "profile") -> str:
    return f"{profile_name}.coords[{json.dumps(coord_name)}].values.item()"


def attr_value_expression(attr_name: str, *, profile_name: str = "profile") -> str:
    return f"{profile_name}.attrs[{json.dumps(attr_name)}]"


def label_context_field_sources(
    contexts: Sequence[Mapping[str, typing.Any]],
) -> dict[str, str]:
    field_sources: dict[str, str] = {}
    for context in contexts:
        sources = context.get(_LABEL_FIELD_SOURCES_KEY, {})
        if not isinstance(sources, dict):
            continue
        for name, source in sources.items():
            if name in field_sources and field_sources[name] != source:
                field_sources[name] = "mixed"
            else:
                field_sources[name] = source
    return field_sources


def string_literal_expression(value: str) -> str:
    return json.dumps(value)


def _format_label_text(
    text: str,
    context: Mapping[str, typing.Any],
    *,
    available_fields: set[str],
    item_name: str,
) -> str:
    parts: list[str] = []
    previous_end = 0
    for match in _PLACEHOLDER_RE.finditer(text):
        parts.append(text[previous_end : match.start()])
        name = match.group("name")
        if name not in context:
            if _placeholder_is_explicit(match):
                raise _unavailable_placeholder_error(
                    match, available_fields, item_name=item_name
                )
            parts.append(match.group(0))
            previous_end = match.end()
            continue
        value = context[name]
        if conversion := match.group("conversion"):
            converters: dict[str, typing.Callable[[typing.Any], str]] = {
                "a": ascii,
                "r": repr,
                "s": str,
            }
            value = converters[conversion](value)
        format_spec = match.group("format") or ""
        try:
            parts.append(format(value, format_spec))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Could not format legend label placeholder {match.group(0)!r} "
                f"for this {item_name}"
            ) from exc
        previous_end = match.end()
    parts.append(text[previous_end:])
    return "".join(parts)


def _unavailable_placeholder_error(
    match: re.Match[str], available_fields: set[str], *, item_name: str
) -> _LabelPlaceholderError:
    placeholder = match.group(0)
    available = _format_available_placeholders(available_fields)
    message = (
        f"Legend label placeholder {placeholder!r} is not available for this "
        f"{item_name}"
    )
    if available:
        message += f". Available placeholders: {available}"
    return _LabelPlaceholderError(message)


def _format_available_placeholders(fields: set[str]) -> str:
    ordered = [name for name in _GENERIC_PLACEHOLDER_ORDER if name in fields] + sorted(
        name for name in fields if name not in _GENERIC_PLACEHOLDER_ORDER
    )
    return ", ".join(f"{{{name}}}" for name in ordered)


def _fstring_literal(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
        .replace("'", "\\'")
        .replace("{", "{{")
        .replace("}", "}}")
    )


def _literal_label_values(
    values: Sequence[str], count: int, *, default: str | None
) -> tuple[str | None, ...]:
    if count < 1:
        return ()
    if not values:
        return (default,) * count
    if len(values) == 1:
        return (values[0],) * count
    if len(values) != count:
        raise ValueError("Legend labels must be one value or one value per profile")
    return tuple(values)


def label_editor_text(operation: FigureOperationState) -> str:
    if operation.line_label_text:
        return operation.line_label_text
    return _format_string_tuple(operation.line_labels)


def default_label_text(
    dim: str | None,
    values: Sequence[typing.Any] = (),
    *,
    fallback: str,
    source_count: int = 1,
) -> str:
    if dim:
        direct_field = f"{{{dim}:g}}" if _values_are_numeric(values) else f"{{{dim}}}"
        label = _property_label_template(dim, direct_field)
    else:
        label = fallback
    if source_count > 1:
        return f"{{source}}, {label}"
    return label


def _property_label_template(dim: str, value_field: str) -> str:
    name = plot_annotations.name_for_dim(dim, escaped=False)
    unit = plot_annotations.unit_for_dim(dim)
    unit_text = f" {unit}" if unit else ""
    if name:
        return f"${name} = {value_field}${unit_text}"
    return f"${value_field}${unit_text}"


def _values_are_numeric(values: Sequence[typing.Any]) -> bool:
    value_list = list(values)
    if not value_list:
        return False
    for value in value_list:
        try:
            float(value)
        except (TypeError, ValueError):
            return False
    return True


def label_text_tooltip(
    contexts: Sequence[Mapping[str, typing.Any]],
    *,
    item_name: str,
) -> str:
    available = _format_available_placeholders(_available_field_names(contexts))
    if not available:
        return f"Enter one label or comma-separated {item_name} labels."
    return (
        f"Enter labels or placeholders. Available: {available}.\n"
        "Numeric placeholders can use formats such as {value:g}.\n"
        "Plain text and LaTeX braces are kept as typed."
    )


def update_current_line_label_text(tool: FigureComposerTool, text: str) -> None:
    editable = tool._editable_operations()
    if not editable:
        return
    selected_ids = {operation.operation_id for _index, operation in editable}
    original_operations = {
        operation.operation_id: operation for _index, operation in editable
    }
    operations = list(tool._recipe.operations)
    newly_labeled_groups: dict[
        tuple[tuple[tuple[int, int], ...], tuple[str, ...], str],
        tuple[int, FigureOperationState],
    ] = {}
    changed = False
    preview_affected = False
    labels_active = bool(text.strip())
    for index, operation in enumerate(tuple(operations)):
        if operation.operation_id not in selected_ids:
            continue
        if operation.kind not in {
            FigureOperationKind.LINE,
            FigureOperationKind.PLOT_SLICES,
        }:
            continue
        updated = operation.model_copy(
            update={"line_label_text": text, "line_labels": ()}
        )
        operation_changed = updated != operation
        changed = changed or operation_changed
        if operation_changed and tool._operation_change_affects_preview(
            operation, updated
        ):
            preview_affected = True
        operations[index] = updated
        original_operation = original_operations.get(operation.operation_id)
        if (
            original_operation is None
            or not labels_active
            or original_operation.line_label_text
            or original_operation.line_labels
            or not updated.enabled
        ):
            continue
        key = _line_axes_key(updated)
        previous = newly_labeled_groups.get(key)
        if previous is None or index > previous[0]:
            newly_labeled_groups[key] = (index, updated)

    for index, operation in sorted(
        newly_labeled_groups.values(), key=lambda item: item[0], reverse=True
    ):
        legend_operation = FigureOperationState.method(
            family=FigureMethodFamily.AXES,
            name="legend",
            label="Legend",
            axes=operation.axes.model_copy(deep=True),
        )
        if not _has_later_legend_step(operations, index, legend_operation):
            operations.insert(index + 1, legend_operation)
            changed = True
            preview_affected = True
    if not changed:
        return
    tool._recipe = tool._recipe.model_copy(update={"operations": tuple(operations)})
    tool._refresh_operation_list()
    tool._sync_axes_selector()
    tool._update_step_action_buttons()
    tool._refresh_step_section_button_texts()
    current = tool._current_operation()
    tool._update_source_status(current[1] if current is not None else None)
    tool._notify_operation_changed(preview_affected=preview_affected)
    tool._write_state()


def _line_axes_key(
    operation: FigureOperationState,
) -> tuple[tuple[tuple[int, int], ...], tuple[str, ...], str]:
    return operation.axes.axes, operation.axes.axes_ids, operation.axes.expression


def _has_later_legend_step(
    operations: list[FigureOperationState],
    index: int,
    operation: FigureOperationState,
) -> bool:
    axes = operation.axes.model_dump()
    return any(
        later.kind == FigureOperationKind.METHOD
        and later.method_family == FigureMethodFamily.AXES
        and later.method_name == "legend"
        and later.axes.model_dump() == axes
        for later in operations[index + 1 :]
    )
