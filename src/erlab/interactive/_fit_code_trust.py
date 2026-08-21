"""Extract executable content from serialized lmfit data without decoding it.

These helpers inspect JSON and NetCDF text. They never import a module named by a
document and never deserialize a saved callable. Callables that resolve to exact,
locally known library symbols produce no trust entry. Unknown callables,
``ExpressionModel`` code, init scripts, and saved parameter expressions produce an
entry for the shared authorization policy.
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
    from collections.abc import Iterable, Iterator


def _detail(
    kind: str, location: str, code: str, **payload: typing.Any
) -> dict[str, typing.Any]:
    return {"code": code, "kind": kind, "location": location, **payload}


def _walk_serialized(
    value: typing.Any, path: str = "root"
) -> Iterator[tuple[str, typing.Any, str | None]]:
    """Yield parsed lmfit values and malformed nested parameter strings."""
    yield path, value, None
    if isinstance(value, dict):
        if value.get("__class__") == "Callable":
            return
        for key in sorted(value):
            item = value[key]
            child_path = f"{path}/{key}"
            if key in {"params", "init_params"} and isinstance(item, str):
                try:
                    item = json.loads(item)
                except (TypeError, ValueError):
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
    value: typing.Any, path: str = "root"
) -> list[dict[str, typing.Any]]:
    details: list[dict[str, typing.Any]] = []
    for item_path, item, malformed in _walk_serialized(value, path):
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
        if not isinstance(item, dict):
            continue
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
            if isinstance(code, str) and code.strip():
                details.append(_detail(kind, f"{item_path}/{key}", code))

        params = item.get("params")
        if isinstance(params, list):
            for index, state in enumerate(params):
                if (
                    isinstance(state, list | tuple)
                    and len(state) >= 4
                    and isinstance(state[3], str)
                    and state[3].strip()
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
    serialized: str, path: str = "root"
) -> list[dict[str, typing.Any]]:
    try:
        value = json.loads(serialized)
    except (TypeError, ValueError):
        return [
            _detail(
                "unknown-serialized-content",
                path,
                "Invalid serialized lmfit content",
                serialized_sha256=hashlib.sha256(serialized.encode()).hexdigest(),
            )
        ]
    return _executable_details(value, path)


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
):
    """Return an entry only when a serialized model can run non-library code."""
    details = (
        _canonical_model_details(_serialized_lmfit_details(serialized_model))
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
        if isinstance(expression, str) and expression.strip()
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
