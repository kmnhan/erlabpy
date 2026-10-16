"""Extract executable content from serialized lmfit data without decoding it.

These helpers inspect JSON and NetCDF text. They never import a module named by a
document and never deserialize a saved callable. Callables that resolve to exact,
locally known library symbols and declarative built-in models produce no trust entry.
Unknown callables, ``ExpressionModel`` code, init scripts, and saved parameter
expressions produce an entry for the shared authorization policy.
"""

from __future__ import annotations

import functools
import hashlib
import io
import json
import typing
import urllib.parse

from erlab.interactive._code_trust import create_entry, create_payload_entry

if typing.TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping


_ModelExpressions = tuple[tuple[str, str], ...]
_ModelDefaultsMemo = dict[str, _ModelExpressions | None]
_SerializedItem = tuple[str, dict[str, typing.Any] | None, str | None]


def _detail(
    kind: str, location: str, code: str, **payload: typing.Any
) -> dict[str, typing.Any]:
    return {"code": code, "kind": kind, "location": location, **payload}


def _walk_serialized(
    value: typing.Any, path: str = "root"
) -> Iterator[_SerializedItem]:
    """Yield lmfit dictionaries and malformed nested parameter strings."""
    if isinstance(value, dict):
        yield path, value, None
        if value.get("__class__") == "Callable":
            return
        for key in sorted(value):
            item = value[key]
            child_path = f"{path}/{key}"
            if key in {"params", "init_params"} and isinstance(item, str):
                try:
                    item = json.loads(item)
                except ValueError:
                    yield child_path, None, value[key]
                    continue
            yield from _walk_serialized(item, child_path)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk_serialized(item, f"{path}/{index}")


def _callable_references(value: typing.Any) -> set[tuple[str, str]]:
    references = set()
    for _path, item, malformed in _walk_serialized(value):
        if malformed is not None or not isinstance(item, dict):
            continue
        importer = item.get("importer")
        name = item.get("__name__")
        if (
            item.get("__class__") == "Callable"
            and isinstance(importer, str)
            and isinstance(name, str)
        ):
            references.add((importer, name))
    return references


def lmfit_model_default_parameter_expressions(model: typing.Any) -> dict[str, str]:
    """Return constraints from a model constructed by installed library code."""
    return {
        str(name): expression
        for name, param in model.make_params().items()
        if isinstance((expression := param._expr), str) and expression.strip()
    }


def _builtin_model_defaults(
    value: typing.Any,
    serialized_items: Iterable[_SerializedItem],
    *,
    model_reference: str | None,
    model_defaults_memo: _ModelDefaultsMemo | None,
) -> tuple[dict[str, _ModelExpressions], list[dict[str, typing.Any]]]:
    from erlab.analysis.fit import _serialization

    matches: dict[str, _ModelExpressions] = {}
    details: list[dict[str, typing.Any]] = []
    reference_matches = True
    if model_reference:
        try:
            reference = _serialization.inspect_model_reference(value)
        except (TypeError, ValueError):
            reference = None
        reference_matches = model_reference == reference
    for item_path, item, malformed in serialized_items:
        if malformed is not None or item is None or "funcdef" not in item:
            continue
        serialized_item = json.dumps(item, sort_keys=True)
        expressions = None
        error = None
        try:
            if (
                model_defaults_memo is not None
                and serialized_item in model_defaults_memo
            ):
                expressions = model_defaults_memo[serialized_item]
            else:
                model = _serialization.inspect_descriptor(item)
                if model is not None:
                    expressions = tuple(
                        lmfit_model_default_parameter_expressions(model).items()
                    )
                if model_defaults_memo is not None:
                    model_defaults_memo[serialized_item] = expressions
        except (TypeError, ValueError) as exc:
            error = str(exc)
        if error is None:
            if expressions is None:
                continue
            if reference_matches:
                matches[item_path] = expressions
                continue
            error = "Saved model class does not match its definition"
        details.append(
            _detail(
                "unrecognized-builtin-model",
                item_path,
                error,
                serialized_model=item,
            )
        )
    return matches, details


@functools.cache
def _safe_library_callable_references() -> frozenset[tuple[str, str]]:
    """Return exact importable callable references emitted by local lmfit."""
    import lmfit
    import lmfit.jsonutils

    import erlab.analysis.fit.functions

    references = _callable_references(json.loads(lmfit.Parameters().dumps()))
    for module, names in (
        (lmfit.lineshapes, lmfit.lineshapes.functions),
        (erlab.analysis.fit.functions, erlab.analysis.fit.functions.__all__),
    ):
        for name in names:
            function = getattr(module, name, None)
            if not callable(function):
                continue
            importer = lmfit.jsonutils.find_importer(function)
            if isinstance(importer, str):
                references.add((importer, str(function.__name__)))
    return frozenset(references)


def _executable_details(
    value: typing.Any,
    path: str = "root",
    *,
    model_reference: str | None = None,
    model_defaults_memo: _ModelDefaultsMemo | None = None,
) -> list[dict[str, typing.Any]]:
    details: list[dict[str, typing.Any]] = []
    serialized_items = tuple(_walk_serialized(value, path))
    safe_models, descriptor_details = _builtin_model_defaults(
        value,
        serialized_items,
        model_reference=model_reference,
        model_defaults_memo=model_defaults_memo,
    )
    details.extend(descriptor_details)
    safe_parameter_expressions = {
        expression for expressions in safe_models.values() for expression in expressions
    }
    safe_hint_locations: set[str] = set()
    for item_path, item, _malformed in serialized_items:
        if item_path not in safe_models or item is None:
            continue
        defaults = dict(safe_models[item_path])
        for name, hint in item.get("param_hints", {}).items():
            if (
                isinstance(hint, dict)
                and isinstance(hint.get("expr"), str)
                and defaults.get(f"{item['prefix']}{name}") == hint["expr"]
            ):
                safe_hint_locations.add(f"{item_path}/param_hints/{name}/expr")
    for item_path, item, malformed in serialized_items:
        if any(
            item_path == f"{safe_path}/funcdef"
            or item_path.startswith(f"{safe_path}/funcdef/")
            for safe_path in safe_models
        ):
            continue
        if malformed is not None:
            details.append(
                _detail(
                    "unknown-serialized-content",
                    item_path,
                    "Invalid serialized lmfit parameters",
                    serialized_sha256=hashlib.sha256(malformed.encode()).hexdigest(),
                )
            )
            continue
        item = typing.cast("dict[str, typing.Any]", item)
        if item.get("__class__") == "Callable":
            importer = item.get("importer")
            name = item.get("__name__")
            if (
                (importer, name)
                if isinstance(importer, str) and isinstance(name, str)
                else None
            ) not in _safe_library_callable_references():
                details.append(
                    _detail(
                        "serialized-callable",
                        item_path,
                        f"{importer or '<embedded>'}:{name or '<unknown>'}",
                        serialized_callable=item,
                    )
                )
            continue

        function_definition = item.get("funcdef")
        if item.get("funcname") == "_eval" and isinstance(function_definition, str):
            details.append(
                _detail(
                    "expression-model",
                    f"{item_path}/funcdef",
                    function_definition,
                )
            )

        for key, kind in (
            ("init_script", "expression-model-init-script"),
            ("expr", "parameter-expression"),
        ):
            code = item.get(key)
            if (
                isinstance(code, str)
                and code.strip()
                and f"{item_path}/{key}" not in safe_hint_locations
            ):
                details.append(_detail(kind, f"{item_path}/{key}", code))

        params = item.get("params")
        if isinstance(params, list):
            for index, state in enumerate(params):
                if (
                    isinstance(state, list | tuple)
                    and len(state) >= 4
                    and isinstance(state[3], str)
                    and state[3].strip()
                    and (str(state[0]), state[3]) not in safe_parameter_expressions
                ):
                    details.append(
                        _detail(
                            "parameter-expression",
                            f"{item_path}/params/{index}:{state[0]}",
                            state[3],
                        )
                    )
    return details


def _serialized_lmfit_details(
    serialized: str,
    path: str = "root",
    *,
    model_reference: str | None = None,
    model_defaults_memo: _ModelDefaultsMemo | None = None,
) -> list[dict[str, typing.Any]]:
    try:
        value = json.loads(serialized)
    except ValueError:
        return [
            _detail(
                "unknown-serialized-content",
                path,
                "Invalid serialized lmfit content",
                serialized_sha256=hashlib.sha256(serialized.encode()).hexdigest(),
            )
        ]
    return _executable_details(
        value,
        path,
        model_reference=model_reference,
        model_defaults_memo=model_defaults_memo,
    )


def lmfit_model_safe_parameter_expressions(
    serialized_model: str,
    model_reference: str,
) -> dict[str, str]:
    """Return default constraints from declarative installed model definitions."""
    try:
        value = json.loads(serialized_model)
    except ValueError:
        return {}
    matches, details = _builtin_model_defaults(
        value,
        _walk_serialized(value),
        model_reference=model_reference,
        model_defaults_memo=None,
    )
    if details:
        return {}
    return dict(
        expression for expressions in matches.values() for expression in expressions
    )


def _entry(
    details: list[dict[str, typing.Any]],
    *,
    feature: str,
    location: str,
    malformed_fallback: bytes,
    context: dict[str, typing.Any] | None = None,
):
    if not details:
        return None
    review = [
        {key: str(detail[key]) for key in ("code", "kind", "location")}
        for detail in details
    ]
    try:
        executable_payload = json.dumps(
            details,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    except (TypeError, ValueError):
        executable_payload = (
            b"malformed-lmfit-payload:" + hashlib.sha256(malformed_fallback).digest()
        )
    return create_payload_entry(
        feature,
        location,
        "\n\n".join(
            f"# {item['kind']} at {item['location']}\n{item['code']}" for item in review
        ),
        executable_payload,
        {**(context or {}), "executable_content": review},
    )


def _canonical_model_details(
    details: Iterable[dict[str, typing.Any]],
) -> list[dict[str, typing.Any]]:
    """Remove lmfit JSON layout details from executable model identities."""
    counts: dict[str, int] = {}
    canonical = []
    for detail in details:
        kind = str(detail["kind"])
        index = counts.get(kind, 0)
        counts[kind] = index + 1
        canonical.append({**detail, "location": f"{kind}/{index}"})
    return canonical


def lmfit_model_code_entry(
    serialized_model: str,
    *,
    feature: str,
    location: str,
    model_reference: str | None = None,
):
    """Return an entry only when a serialized model can run non-library code."""
    details = (
        _canonical_model_details(
            _serialized_lmfit_details(serialized_model, model_reference=model_reference)
        )
        if serialized_model
        else []
    )
    return _entry(
        details,
        feature=feature,
        location=location,
        malformed_fallback=serialized_model.encode(),
    )


def lmfit_expression_model_code_entries(
    expression: str,
    init_script: str | None,
    *,
    feature: str,
    location: str,
):
    """Return exact entries for one locally authored ExpressionModel edit."""
    model_entry = _entry(
        _canonical_model_details(
            [_detail("expression-model", "expression", expression)]
        ),
        feature=feature,
        location=location,
        malformed_fallback=expression.encode(),
    )
    entries = [] if model_entry is None else [model_entry]
    if init_script:
        entries.append(
            create_entry(
                feature,
                f"{location}/init-script",
                init_script,
                {"kind": "expression-model-init-script"},
            )
        )
    return tuple(entries)


def lmfit_parameter_expression_entries(
    expressions: Iterable[tuple[typing.Any, typing.Any]] | None,
    *,
    feature: str,
    location_prefix: str,
    safe_expressions: Mapping[str, str] | None = None,
):
    """Return entries for saved ``(parameter name, expression)`` pairs."""
    if expressions is None:
        return ()
    return tuple(
        create_entry(
            feature,
            f"{location_prefix}/{urllib.parse.quote(str(name), safe='')}",
            expression,
            {"parameter": name},
        )
        for name, expression in expressions
        if isinstance(expression, str)
        and expression.strip()
        and (safe_expressions is None or safe_expressions.get(str(name)) != expression)
    )


def lmfit_result_code_entry(
    payload: bytes,
    *,
    feature: str,
    location: str,
):
    """Return an entry only when an xarray-lmfit payload can run custom code."""
    import numpy as np
    import xarray as xr

    details: list[dict[str, typing.Any]] = []
    invalid = False
    model_defaults_memo: _ModelDefaultsMemo = {}
    try:
        with xr.open_dataset(io.BytesIO(payload), engine="h5netcdf") as dataset:
            result_names = sorted(
                (
                    name
                    for name in dataset.data_vars
                    if str(name).endswith("modelfit_results")
                ),
                key=str,
            )
            if not result_names:
                invalid = True
            for name in result_names:
                for index, serialized in enumerate(
                    np.asarray(dataset[name].values).flat
                ):
                    if isinstance(serialized, bytes):
                        serialized = serialized.decode()
                    if not isinstance(serialized, str):
                        invalid = True
                        continue
                    details.extend(
                        _serialized_lmfit_details(
                            serialized,
                            path=f"{name}/{index}/root",
                            model_defaults_memo=model_defaults_memo,
                        )
                    )
    except Exception:
        invalid = True
    if invalid:
        details = [
            _detail(
                "unknown-serialized-content",
                "root",
                "Unrecognized serialized lmfit result",
                serialized_sha256=hashlib.sha256(payload).hexdigest(),
            )
        ]

    return _entry(
        details,
        feature=feature,
        location=location,
        malformed_fallback=payload,
    )
