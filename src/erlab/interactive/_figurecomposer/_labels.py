"""Legend label helpers for line-like Figure Composer operations."""

from __future__ import annotations

import ast
import json
import keyword
import re
import typing

import numpy as np

import erlab.plotting.annotations as plot_annotations
from erlab.interactive._figurecomposer._model._state import (
    FigureMethodFamily,
    FigureOperationKind,
    FigureOperationState,
)
from erlab.interactive._figurecomposer._text import (
    _format_string_tuple,
    _string_tuple_from_text,
)

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence

    import xarray as xr


_GENERIC_PLACEHOLDER_ORDER = ("value", "dim", "number", "index", "source")
_LABEL_FIELD_SOURCES_KEY = "__figure_composer_label_field_sources__"
_LABEL_FIELD_ORIGINAL_NAMES_KEY = "__figure_composer_label_original_names__"
_LABEL_COORD_ALIASES_KEY = "__figure_composer_label_coord_aliases__"
_LABEL_ATTR_ALIASES_KEY = "__figure_composer_label_attr_aliases__"
_PLACEHOLDER_HELP_ERROR_SUFFIX = ". See Legend labels help for valid names"
_LABEL_INTERNAL_KEYS = frozenset(
    {
        _LABEL_FIELD_SOURCES_KEY,
        _LABEL_FIELD_ORIGINAL_NAMES_KEY,
        _LABEL_COORD_ALIASES_KEY,
        _LABEL_ATTR_ALIASES_KEY,
    }
)
_FSTRING_ALLOWED_EXPR_NODES = (
    ast.Add,
    ast.And,
    ast.BinOp,
    ast.BoolOp,
    ast.Compare,
    ast.Constant,
    ast.Div,
    ast.Eq,
    ast.FloorDiv,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.Is,
    ast.IsNot,
    ast.Load,
    ast.Lt,
    ast.LtE,
    ast.Mod,
    ast.Mult,
    ast.Name,
    ast.Not,
    ast.NotEq,
    ast.NotIn,
    ast.Or,
    ast.Pow,
    ast.Sub,
    ast.UAdd,
    ast.USub,
    ast.UnaryOp,
)


class _ParsedLabelField(typing.NamedTuple):
    placeholder: str
    expression: ast.expr
    conversion: int
    format_spec: ast.JoinedStr | None


class _LabelPlaceholderError(ValueError):
    """Raised when a label placeholder is explicit but unavailable."""


def _field_names(text: str) -> set[str]:
    names: set[str] = set()
    for chunk in _iter_label_chunks(text):
        if isinstance(chunk, _ParsedLabelField):
            names.update(_field_names_from_parsed_field(chunk))
    return names


def _explicit_field_names(text: str) -> set[str]:
    names: set[str] = set()
    for chunk in _iter_label_chunks(text):
        if isinstance(chunk, _ParsedLabelField) and _placeholder_is_explicit(chunk):
            names.update(_field_names_from_parsed_field(chunk))
    return names


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


def _label_identifier(name: str) -> str:
    identifier = re.sub(r"\W+", "_", str(name).strip(), flags=re.UNICODE).strip("_")
    if not identifier:
        identifier = "field"
    if identifier[0].isdigit():
        identifier = f"_{identifier}"
    if keyword.iskeyword(identifier):
        identifier = f"{identifier}_"
    if not identifier.isidentifier():
        identifier = "field"
    return identifier


def _unique_label_identifier(name: str, used: set[str]) -> str:
    base = _label_identifier(name)
    if base not in used:
        return base
    index = 2
    while f"{base}_{index}" in used:
        index += 1
    return f"{base}_{index}"


def label_coord_placeholder_name(coord_name: str) -> str:
    return _unique_label_identifier(coord_name, set(_GENERIC_PLACEHOLDER_ORDER))


def _add_label_field(
    context: dict[str, typing.Any],
    *,
    original_name: str,
    value: typing.Any,
    source: typing.Literal["coord", "attr"],
    used: set[str],
    field_sources: dict[str, str],
    original_names: dict[str, str],
    coord_aliases: dict[str, str],
    attr_aliases: dict[str, str],
) -> str:
    alias = _unique_label_identifier(original_name, used)
    used.add(alias)
    context[alias] = value
    field_sources[alias] = source
    original_names[alias] = original_name
    if source == "coord":
        coord_aliases.setdefault(original_name, alias)
    else:
        attr_aliases.setdefault(original_name, alias)
    return alias


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
    used = set(context)
    if source is not None:
        context["source"] = source
        used.add("source")
    if dim is not None:
        context["dim"] = dim
        used.add("dim")
    if value is not None:
        scalar = _scalar_value(value)
        context["value"] = value if scalar is None else scalar
        used.add("value")
    field_sources: dict[str, str] = {}
    original_names: dict[str, str] = {}
    coord_aliases: dict[str, str] = {}
    attr_aliases: dict[str, str] = {}
    if profile is not None:
        for name, coord in profile.coords.items():
            scalar = _scalar_value(coord.values)
            if scalar is not None:
                _add_label_field(
                    context,
                    original_name=str(name),
                    value=scalar,
                    source="coord",
                    used=used,
                    field_sources=field_sources,
                    original_names=original_names,
                    coord_aliases=coord_aliases,
                    attr_aliases=attr_aliases,
                )
        for name, value in profile.attrs.items():
            scalar = _scalar_value(value)
            if scalar is not None:
                _add_label_field(
                    context,
                    original_name=str(name),
                    value=scalar,
                    source="attr",
                    used=used,
                    field_sources=field_sources,
                    original_names=original_names,
                    coord_aliases=coord_aliases,
                    attr_aliases=attr_aliases,
                )
    if dim is not None and value is not None and dim not in coord_aliases:
        scalar = _scalar_value(value)
        if scalar is not None:
            _add_label_field(
                context,
                original_name=dim,
                value=scalar,
                source="coord",
                used=used,
                field_sources=field_sources,
                original_names=original_names,
                coord_aliases=coord_aliases,
                attr_aliases=attr_aliases,
            )
    if field_sources:
        context[_LABEL_FIELD_SOURCES_KEY] = field_sources
        context[_LABEL_FIELD_ORIGINAL_NAMES_KEY] = original_names
        context[_LABEL_COORD_ALIASES_KEY] = coord_aliases
        context[_LABEL_ATTR_ALIASES_KEY] = attr_aliases
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

    available_fields = _available_field_names(contexts)
    if _text_uses_placeholders(text, available_fields=available_fields):
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
                    f"Could not format legend label {text!r} for this {item_name}: "
                    f"{exc}"
                ) from exc
        return tuple(labels)

    return _literal_label_values(_string_tuple_from_text(text), count, default=default)


def label_text_uses_placeholders(
    text: str, contexts: Sequence[Mapping[str, typing.Any]]
) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    available_fields = _available_field_names(contexts)
    return _text_uses_placeholders(text, available_fields=available_fields)


def label_field_names(text: str) -> set[str]:
    return _field_names(text)


def label_fstring_code(text: str, field_expressions: Mapping[str, str]) -> str:
    parts = ['f"']
    for chunk in _iter_label_chunks(text):
        if isinstance(chunk, str):
            parts.append(_fstring_literal(chunk))
            continue
        fields = _field_names_from_parsed_field(chunk)
        if not fields and _placeholder_is_literal_when_missing(chunk):
            parts.append(_fstring_literal(chunk.placeholder))
            continue
        missing = fields - set(field_expressions)
        if missing:
            if _placeholder_is_literal_when_missing(chunk):
                parts.append(_fstring_literal(chunk.placeholder))
                continue
            raise _unavailable_placeholder_error(
                chunk.placeholder,
                set(field_expressions),
                item_name="generated code",
            )
        parts.append(_field_fstring_code(chunk, field_expressions))
    parts.append('"')
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


def label_context_original_field_names(
    contexts: Sequence[Mapping[str, typing.Any]],
) -> dict[str, str]:
    original_names: dict[str, str] = {}
    for context in contexts:
        names = context.get(_LABEL_FIELD_ORIGINAL_NAMES_KEY, {})
        if not isinstance(names, dict):
            continue
        for alias, original_name in names.items():
            original = str(original_name)
            if alias in original_names and original_names[alias] != original:
                original_names[alias] = ""
            else:
                original_names[alias] = original
    return original_names


def label_context_coord_alias(
    context: Mapping[str, typing.Any], coord_name: str
) -> str | None:
    aliases = context.get(_LABEL_COORD_ALIASES_KEY, {})
    if isinstance(aliases, dict):
        alias = aliases.get(coord_name)
        if isinstance(alias, str):
            return alias
    if coord_name in context:
        return coord_name
    return None


def string_literal_expression(value: str) -> str:
    return json.dumps(value)


def _text_uses_placeholders(text: str, *, available_fields: set[str]) -> bool:
    for chunk in _iter_label_chunks(text, available_fields=available_fields):
        if not isinstance(chunk, _ParsedLabelField):
            continue
        fields = _field_names_from_parsed_field(chunk)
        if fields & available_fields or _placeholder_is_explicit(chunk):
            return True
    return False


def _iter_label_chunks(
    text: str,
    *,
    available_fields: set[str] | None = None,
    item_name: str = "profile",
) -> Iterator[str | _ParsedLabelField]:
    literal_parts: list[str] = []
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if char == "{" and index + 1 < length and text[index + 1] == "{":
            literal_parts.append("{")
            index += 2
            continue
        if char == "}" and index + 1 < length and text[index + 1] == "}":
            literal_parts.append("}")
            index += 2
            continue
        if char != "{":
            literal_parts.append(char)
            index += 1
            continue
        end = _find_label_placeholder_end(text, index)
        if end < 0:
            literal_parts.append(text[index:])
            break
        if literal_parts:
            yield "".join(literal_parts)
            literal_parts.clear()
        placeholder = text[index : end + 1]
        field_text = text[index + 1 : end]
        parsed = _parse_label_field(
            placeholder,
            field_text,
            available_fields=available_fields or set(),
            item_name=item_name,
        )
        if parsed is None:
            yield placeholder
        else:
            yield parsed
        index = end + 1
    if literal_parts:
        yield "".join(literal_parts)


def _find_label_placeholder_end(text: str, start: int) -> int:
    depth = 0
    index = start + 1
    while index < len(text):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            if depth == 0:
                return index
            depth -= 1
        index += 1
    return -1


def _parse_label_field(
    placeholder: str,
    field_text: str,
    *,
    available_fields: set[str],
    item_name: str,
) -> _ParsedLabelField | None:
    if not field_text:
        return None
    try:
        expression = ast.parse("f" + repr(placeholder), mode="eval")
    except SyntaxError as exc:
        if _invalid_placeholder_is_literal(field_text, available_fields):
            return None
        raise _invalid_placeholder_error(
            placeholder, field_text, available_fields, item_name=item_name
        ) from exc
    if not isinstance(expression.body, ast.JoinedStr):
        return None
    fields = [
        value
        for value in expression.body.values
        if isinstance(value, ast.FormattedValue)
    ]
    if len(fields) != 1:
        return None
    field = fields[0]
    format_spec = field.format_spec
    if format_spec is not None and not isinstance(format_spec, ast.JoinedStr):
        return None
    return _ParsedLabelField(
        placeholder=placeholder,
        expression=field.value,
        conversion=field.conversion,
        format_spec=format_spec,
    )


def _invalid_placeholder_is_literal(
    field_text: str, available_fields: set[str]
) -> bool:
    if ":" in field_text or "!" in field_text:
        return False
    normalized = _label_identifier(field_text)
    return normalized not in available_fields


def _invalid_placeholder_error(
    placeholder: str,
    field_text: str,
    available_fields: set[str],
    *,
    item_name: str,
) -> _LabelPlaceholderError:
    message = (
        f"Legend label placeholder {placeholder!r} is not valid f-string syntax "
        f"for this {item_name}"
    )
    normalized = _label_identifier(field_text.split("!", 1)[0].split(":", 1)[0])
    if normalized in available_fields:
        suffix = field_text[len(field_text.split("!", 1)[0].split(":", 1)[0]) :]
        message += f". Use {{{normalized}{suffix}}} instead"
    if available_fields:
        message += _PLACEHOLDER_HELP_ERROR_SUFFIX
    return _LabelPlaceholderError(message)


def _field_names_from_parsed_field(field: _ParsedLabelField) -> set[str]:
    names = _field_names_from_expression(field.expression)
    if field.format_spec is not None:
        names.update(_field_names_from_joined_str(field.format_spec))
    return names


def _field_names_from_joined_str(joined: ast.JoinedStr) -> set[str]:
    names: set[str] = set()
    for value in joined.values:
        if isinstance(value, ast.FormattedValue):
            names.update(_field_names_from_expression(value.value))
            if isinstance(value.format_spec, ast.JoinedStr):
                names.update(_field_names_from_joined_str(value.format_spec))
    return names


def _field_names_from_expression(expression: ast.AST) -> set[str]:
    return {node.id for node in ast.walk(expression) if isinstance(node, ast.Name)}


def _placeholder_is_explicit(field: _ParsedLabelField) -> bool:
    return not _placeholder_is_literal_when_missing(field)


def _placeholder_is_literal_when_missing(field: _ParsedLabelField) -> bool:
    return (
        (
            isinstance(field.expression, ast.Name)
            or not _field_names_from_expression(field.expression)
        )
        and field.conversion == -1
        and field.format_spec is None
    )


def _format_label_text(
    text: str,
    context: Mapping[str, typing.Any],
    *,
    available_fields: set[str],
    item_name: str,
) -> str:
    parts: list[str] = []
    for chunk in _iter_label_chunks(
        text, available_fields=available_fields, item_name=item_name
    ):
        if isinstance(chunk, str):
            parts.append(chunk)
            continue
        if not _field_names_from_parsed_field(
            chunk
        ) and _placeholder_is_literal_when_missing(chunk):
            parts.append(chunk.placeholder)
            continue
        try:
            parts.append(_format_label_field(chunk, context, item_name=item_name))
        except NameError:
            if _placeholder_is_literal_when_missing(chunk):
                parts.append(chunk.placeholder)
                continue
            raise _unavailable_placeholder_error(
                chunk.placeholder, available_fields, item_name=item_name
            ) from None
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Could not format legend label placeholder {chunk.placeholder!r} "
                f"for this {item_name}: {exc}"
            ) from exc
    return "".join(parts)


def _format_label_field(
    field: _ParsedLabelField,
    context: Mapping[str, typing.Any],
    *,
    item_name: str,
) -> str:
    _validate_label_expression(field.expression, item_name=item_name)
    value = _evaluate_label_expression(field.expression, context)
    if field.conversion != -1:
        converters: dict[int, Callable[[typing.Any], str]] = {
            ord("a"): ascii,
            ord("r"): repr,
            ord("s"): str,
        }
        if field.conversion not in converters:
            raise ValueError(f"Unsupported f-string conversion {field.conversion!r}")
        value = converters[field.conversion](value)
    format_spec = ""
    if field.format_spec is not None:
        format_spec = _evaluate_format_spec(
            field.format_spec, context, item_name=item_name
        )
    return format(value, format_spec)


def _evaluate_format_spec(
    joined: ast.JoinedStr,
    context: Mapping[str, typing.Any],
    *,
    item_name: str,
) -> str:
    parts: list[str] = []
    for value in joined.values:
        if isinstance(value, ast.Constant):
            parts.append(str(value.value))
        elif isinstance(value, ast.FormattedValue):
            nested = _ParsedLabelField(
                placeholder="format spec",
                expression=value.value,
                conversion=value.conversion,
                format_spec=value.format_spec
                if isinstance(value.format_spec, ast.JoinedStr)
                else None,
            )
            parts.append(_format_label_field(nested, context, item_name=item_name))
    return "".join(parts)


def _validate_label_expression(expression: ast.AST, *, item_name: str) -> None:
    for node in ast.walk(expression):
        if isinstance(node, _FSTRING_ALLOWED_EXPR_NODES):
            continue
        raise ValueError(
            "Legend label expressions support scalar placeholders and basic "
            f"operators only for this {item_name}"
        )


def _evaluate_label_expression(
    expression: ast.expr, context: Mapping[str, typing.Any]
) -> typing.Any:
    compiled = compile(
        ast.Expression(expression),
        "<figure-composer-label>",
        "eval",
    )
    return eval(compiled, {"__builtins__": {}}, dict(context))  # noqa: S307


def _unavailable_placeholder_error(
    placeholder: str, available_fields: set[str], *, item_name: str
) -> _LabelPlaceholderError:
    message = (
        f"Legend label placeholder {placeholder!r} is not available for this "
        f"{item_name}"
    )
    if available_fields:
        message += _PLACEHOLDER_HELP_ERROR_SUFFIX
    return _LabelPlaceholderError(message)


def _field_fstring_code(
    field: _ParsedLabelField, field_expressions: Mapping[str, str]
) -> str:
    _validate_label_expression(field.expression, item_name="generated code")
    expression = _expression_code(field.expression, field_expressions)
    code = "{" + expression
    if field.conversion != -1:
        code += f"!{chr(field.conversion)}"
    if field.format_spec is not None:
        code += ":" + _format_spec_code(field.format_spec, field_expressions)
    return code + "}"


def _expression_code(expression: ast.expr, field_expressions: Mapping[str, str]) -> str:
    transformed = _FieldExpressionSubstituter(field_expressions).visit(
        typing.cast("ast.AST", expression)
    )
    ast.fix_missing_locations(transformed)
    return ast.unparse(transformed)


def _format_spec_code(
    joined: ast.JoinedStr, field_expressions: Mapping[str, str]
) -> str:
    parts: list[str] = []
    for value in joined.values:
        if isinstance(value, ast.Constant):
            parts.append(str(value.value).replace("{", "{{").replace("}", "}}"))
        elif isinstance(value, ast.FormattedValue):
            nested = _ParsedLabelField(
                placeholder="format spec",
                expression=value.value,
                conversion=value.conversion,
                format_spec=value.format_spec
                if isinstance(value.format_spec, ast.JoinedStr)
                else None,
            )
            parts.append(_field_fstring_code(nested, field_expressions))
    return "".join(parts)


class _FieldExpressionSubstituter(ast.NodeTransformer):
    def __init__(self, field_expressions: Mapping[str, str]) -> None:
        super().__init__()
        self._field_expressions = field_expressions

    def visit_Name(self, node: ast.Name) -> ast.AST:
        expression = self._field_expressions.get(node.id)
        if expression is None:
            return node
        replacement = ast.parse(expression, mode="eval").body
        return ast.copy_location(replacement, node)


def _fstring_literal(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
        .replace('"', '\\"')
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
        field = label_coord_placeholder_name(dim)
        direct_field = (
            f"{{{field}:g}}" if _values_are_numeric(values) else f"{{{field}}}"
        )
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
    available_fields = _available_field_names(contexts)
    if not available_fields:
        return (
            f"Enter one label for all {item_name}s, or one comma-separated label "
            f"per {item_name}."
        )
    return (
        "Enter labels directly or use placeholders; click help for examples "
        "and coordinate or attr aliases."
    )


class LabelPlaceholderHelpRow(typing.NamedTuple):
    placeholder: str
    kind: str
    description: str


def label_text_help_placeholder_rows(
    contexts: Sequence[Mapping[str, typing.Any]],
    *,
    item_name: str,
) -> tuple[LabelPlaceholderHelpRow, ...]:
    rows: list[LabelPlaceholderHelpRow] = []
    available_fields = _available_field_names(contexts)
    generic_descriptions = {
        "value": f"current {item_name} coordinate value",
        "dim": "dimension name",
        "number": f"one-based {item_name} number",
        "index": f"zero-based {item_name} index",
        "source": "source name",
    }
    rows.extend(
        LabelPlaceholderHelpRow(
            placeholder=name,
            kind="built-in",
            description=generic_descriptions[name],
        )
        for name in _GENERIC_PLACEHOLDER_ORDER
        if name in available_fields
    )

    field_sources = label_context_field_sources(contexts)
    original_names = label_context_original_field_names(contexts)
    for alias in sorted(original_names):
        original = original_names[alias]
        if not original:
            original = alias
        source = field_sources.get(alias, "field")
        if source == "coord":
            description = f"coordinate {original!r}"
        elif source == "attr":
            description = f"attr {original!r}"
        else:
            description = f"{source} {original!r}"
        rows.append(
            LabelPlaceholderHelpRow(
                placeholder=alias,
                kind=source,
                description=description,
            )
        )
    return tuple(rows)


def operations_with_line_label_text(
    operations: Sequence[FigureOperationState],
    operation_ids: Iterable[str],
    text: str,
) -> tuple[FigureOperationState, ...]:
    """Update selected line labels and add one necessary legend per axes group."""
    selected_ids = set(operation_ids)
    if not selected_ids:
        return tuple(operations)
    original_operations = {
        operation.operation_id: operation
        for operation in operations
        if operation.operation_id in selected_ids
    }
    updated_operations = list(operations)
    newly_labeled_groups: dict[
        tuple[tuple[tuple[int, int], ...], tuple[str, ...], str],
        tuple[int, FigureOperationState],
    ] = {}
    labels_active = bool(text.strip())
    for index, operation in enumerate(tuple(updated_operations)):
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
        updated_operations[index] = updated
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
        if not _has_later_legend_step(updated_operations, index, legend_operation):
            updated_operations.insert(index + 1, legend_operation)
    return tuple(updated_operations)


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
