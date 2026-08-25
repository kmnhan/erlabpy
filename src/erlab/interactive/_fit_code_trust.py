"""Extract executable content from serialized lmfit data without decoding it.

These helpers inspect JSON and NetCDF text. They never import a module named by a
document and never deserialize a saved callable. Callables that resolve to exact,
locally known library symbols and embedded models that exactly match a model produced
by the installed ERLab library produce no trust entry. Unknown callables,
``ExpressionModel`` code, init scripts, and saved parameter expressions produce an
entry for the shared authorization policy.
"""

from __future__ import annotations

import base64
import functools
import hashlib
import io
import json
import pickletools
import re
import typing
import urllib.parse

from erlab.interactive._code_trust import create_entry, create_payload_entry

if typing.TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping


_ModelExpressions = tuple[tuple[str, str], ...]
_ModelMatch = tuple[str, _ModelExpressions]
_ModelMatchKey = tuple[str, str | None]
_ModelMatchMemo = dict[_ModelMatchKey, tuple[dict[str, typing.Any], _ModelMatch | None]]
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


def _wrapped_list(value: typing.Any) -> list[typing.Any] | None:
    if (
        isinstance(value, dict)
        and value.get("__class__") == "List"
        and isinstance(value.get("value"), list)
    ):
        return value["value"]
    return None


def _serialized_model_dict(model: typing.Any) -> dict[str, typing.Any] | None:
    """Return the first serialized lmfit model dictionary produced locally."""
    try:
        value = json.loads(model.dumps())
    except (AttributeError, TypeError, ValueError):
        return None
    for _path, item, malformed in _walk_serialized(value):
        if (
            malformed is None
            and isinstance(item, dict)
            and isinstance(item.get("funcname"), str)
            and isinstance(item.get("funcdef"), dict)
            and "param_hints" in item
        ):
            return item
    return None


def _model_reference(model: typing.Any) -> str:
    model_class = type(model)
    return f"{model_class.__module__}:{model_class.__qualname__}"


def _common_model_kwargs(
    item: dict[str, typing.Any],
) -> dict[str, typing.Any] | None:
    name = item.get("name")
    prefix = item.get("prefix")
    nan_policy = item.get("nan_policy")
    if (
        not isinstance(name, str)
        or not isinstance(prefix, str)
        or nan_policy not in {"raise", "propagate", "omit"}
    ):
        return None
    return {"name": name, "prefix": prefix, "nan_policy": nan_policy}


def _pickle_scalar_after_key(payload: str, key: str) -> int | bool | None:
    """Read one primitive pickle state value without constructing any objects."""
    try:
        operations = pickletools.genops(base64.b64decode(payload, validate=True))
        for operation, argument, _position in operations:
            if operation.name not in {"SHORT_BINUNICODE", "BINUNICODE"}:
                continue
            if argument != key:
                continue
            for value_operation, value_argument, _position in operations:
                if value_operation.name == "MEMOIZE":
                    continue
                if value_operation.name == "NEWTRUE":
                    return True
                if value_operation.name == "NEWFALSE":
                    return False
                if value_operation.name in {"BININT", "BININT1", "BININT2"}:
                    return typing.cast("int", value_argument)
                return None
    except (TypeError, ValueError):
        return None
    return None


def _multipeak_candidate_options(
    item: dict[str, typing.Any],
) -> tuple[int, list[str], bool, str, int, bool, int, bool] | None:
    """Infer bounded declarative options without decoding the callable payload."""
    raw_names = _wrapped_list(item.get("param_root_names"))
    hints = item.get("param_hints")
    funcdef = item.get("funcdef")
    if (
        raw_names is None
        or not isinstance(hints, dict)
        or not isinstance(funcdef, dict)
        or not isinstance((callable_payload := funcdef.get("value")), str)
    ):
        return None
    if not all(isinstance(name, str) for name in raw_names):
        return None
    names = typing.cast("list[str]", raw_names)
    name_set = set(names)
    peak_indices = {
        int(match.group(1))
        for name in names
        if (match := re.fullmatch(r"p(\d+)_.*", name)) is not None
    }
    if not peak_indices:
        return None
    npeaks = max(peak_indices) + 1
    if npeaks > 20 or peak_indices != set(range(npeaks)):
        return None

    peak_shapes: list[str] = []
    for index in range(npeaks):
        label = f"p{index}_"
        roots = {name.removeprefix(label) for name in names if name.startswith(label)}
        if {"sigma", "gamma", "amplitude"}.issubset(roots):
            peak_shapes.append("voigt")
        elif {f"{label}gamma", f"{label}amplitude"}.issubset(hints):
            peak_shapes.append("lorentzian")
        elif {f"{label}sigma", f"{label}amplitude"}.issubset(hints):
            peak_shapes.append("gaussian")
        else:
            return None

    fd_names = {"efermi", "temp", "offset"}
    if name_set.intersection(fd_names) not in (set(), fd_names):
        return None
    fd = fd_names.issubset(name_set)
    convolve = "resolution" in name_set
    oversample = _pickle_scalar_after_key(callable_payload, "oversample")
    segmented = _pickle_scalar_after_key(callable_payload, "segmented")
    if (
        type(oversample) is not int
        or not 1 <= oversample <= 64
        or type(segmented) is not bool
    ):
        return None

    if any(re.fullmatch(r"k_step_\d+", name) for name in names):
        background = "shirley"
        degree = 2
    else:
        coefficients = [
            int(match.group(1))
            for name in names
            if (match := re.fullmatch(r"c(\d+)", name)) is not None
        ]
        if coefficients:
            if set(coefficients) != set(range(max(coefficients) + 1)):
                return None
            background = "polynomial"
            degree = max(coefficients)
            if degree > 10:
                return None
        elif {"const_bkg", "lin_bkg"}.issubset(name_set):
            background = "linear"
            degree = 2
        elif "const_bkg" in name_set:
            background = "constant"
            degree = 2
        else:
            background = "none"
            degree = 2
    return (
        npeaks,
        peak_shapes,
        fd,
        background,
        degree,
        convolve,
        oversample,
        segmented,
    )


def _known_model_candidates(item: dict[str, typing.Any]) -> Iterator[typing.Any]:
    """Yield bounded local models that could have produced one saved dictionary."""
    import erlab.analysis.fit.models

    common = _common_model_kwargs(item)
    funcname = item.get("funcname")
    if common is None or not isinstance(funcname, str):
        return
    if funcname == "MultiPeakFunction":
        options = _multipeak_candidate_options(item)
        if options is None:
            return
        (
            npeaks,
            peak_shapes,
            fd,
            background,
            degree,
            convolve,
            oversample,
            segmented,
        ) = options
        yield erlab.analysis.fit.models.MultiPeakModel(
            npeaks=npeaks,
            peak_shapes=peak_shapes,
            fd=fd,
            background=typing.cast("typing.Any", background),
            degree=degree,
            convolve=convolve,
            oversample=oversample,
            segmented=segmented,
            **common,
        )
        return
    if funcname == "PolynomialFunction":
        names = _wrapped_list(item.get("param_root_names"))
        if names is None:
            return
        coefficients = [
            int(match.group(1))
            for name in names
            if isinstance(name, str)
            and (match := re.fullmatch(r"c(\d+)", name)) is not None
        ]
        if not coefficients or set(coefficients) != set(range(max(coefficients) + 1)):
            return
        degree = max(coefficients)
        if degree <= 20:
            yield erlab.analysis.fit.models.PolynomialModel(degree=degree, **common)
        return
    if funcname == "FermiEdge2dFunction":
        names = _wrapped_list(item.get("param_root_names"))
        if names is None:
            return
        coefficients = [
            int(match.group(1))
            for name in names
            if isinstance(name, str)
            and (match := re.fullmatch(r"c(\d+)", name)) is not None
        ]
        if coefficients and set(coefficients) == set(range(max(coefficients) + 1)):
            degree = max(coefficients)
            if degree <= 20:
                yield erlab.analysis.fit.models.FermiEdge2dModel(
                    degree=degree, **common
                )
        return
    simple_models = {
        "_broadFermiDirac": erlab.analysis.fit.models.FermiDiracModel,
        "_lin_broad_fd": erlab.analysis.fit.models.FermiEdgeModel,
        "_step_linbkg_broad": erlab.analysis.fit.models.StepEdgeModel,
    }
    if (model_class := simple_models.get(funcname)) is not None:
        yield model_class(**common)


def _compute_known_model_match(
    serialized_item: str,
    model_reference: str | None,
) -> _ModelMatch | None:
    """Match exact embedded bytes against a model generated by local library code."""
    try:
        item = json.loads(serialized_item)
    except ValueError:
        return None
    if not isinstance(item, dict):
        return None
    for _attempt in range(2):
        try:
            candidates = _known_model_candidates(item)
            for model in candidates:
                reference = _model_reference(model)
                if model_reference is not None and reference != model_reference:
                    continue
                local_items = [_serialized_model_dict(model)]
                if item.get("funcname") == "MultiPeakFunction":
                    # Evaluation materializes this local cached property. It contains
                    # only information derived from the installed peak functions.
                    _peak_argnames = model.func.peak_argnames
                    local_items.append(_serialized_model_dict(model))
                if item not in local_items:
                    continue
                expressions = tuple(
                    (str(name), expression)
                    for name, param in model.make_params().items()
                    if isinstance((expression := param._expr), str)
                    and expression.strip()
                )
                return reference, expressions
        except Exception:
            return None
        # lmfit serialization can materialize benign callable cache state on the
        # first local comparison. Rebuild the candidates once before failing closed.
    return None


class _NoKnownModelMatch(Exception):
    """Prevent transient failed local comparisons from entering the cache."""


@functools.lru_cache(maxsize=256)
def _cached_known_model_match(
    serialized_item: str,
    model_reference: str | None,
) -> _ModelMatch:
    match = _compute_known_model_match(serialized_item, model_reference)
    if match is None:
        raise _NoKnownModelMatch
    return match


def _known_model_match(
    serialized_item: str,
    model_reference: str | None,
) -> _ModelMatch | None:
    try:
        return _cached_known_model_match(serialized_item, model_reference)
    except _NoKnownModelMatch:
        return None


def _payload_model_match(
    item: dict[str, typing.Any],
    model_reference: str | None,
    memo: _ModelMatchMemo,
) -> _ModelMatch | None:
    """Reuse an exact match only within one serialized result payload."""
    function_definition = item.get("funcdef")
    callable_payload = (
        function_definition.get("value")
        if isinstance(function_definition, dict)
        else None
    )
    key = (
        (callable_payload, model_reference)
        if isinstance(callable_payload, str)
        else None
    )
    if key is not None:
        cached = memo.get(key)
        if cached is not None and item == cached[0]:
            return cached[1]
    else:
        cached = None
    try:
        serialized_item = json.dumps(
            item,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError):
        return None
    if key is None:
        key = (serialized_item, model_reference)
        cached = memo.get(key)
        if cached is not None and item == cached[0]:
            return cached[1]
    match = _known_model_match(serialized_item, model_reference)
    if match is None:
        match = _known_model_match(serialized_item, model_reference)
    if cached is None:
        memo[key] = (item, match)
    return match


def _safe_model_matches(
    value: typing.Any,
    model_reference: str | None = None,
    path: str = "root",
    *,
    model_match_memo: _ModelMatchMemo | None = None,
    serialized_items: Iterable[_SerializedItem] | None = None,
) -> tuple[dict[str, _ModelExpressions], frozenset[tuple[str, str]]]:
    matches: dict[str, _ModelExpressions] = {}
    expressions: set[tuple[str, str]] = set()
    if serialized_items is None:
        serialized_items = _walk_serialized(value, path)
    for item_path, item, malformed in serialized_items:
        if (
            malformed is not None
            or item is None
            or not isinstance(item.get("funcname"), str)
            or not isinstance(item.get("funcdef"), dict)
            or "param_hints" not in item
        ):
            continue
        if model_match_memo is None:
            try:
                serialized_item = json.dumps(
                    item,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            except (TypeError, ValueError):
                continue
            match = _known_model_match(serialized_item, model_reference)
        else:
            match = _payload_model_match(
                item,
                model_reference,
                model_match_memo,
            )
        if match is None:
            continue
        _reference, model_expressions = match
        matches[item_path] = model_expressions
        expressions.update(model_expressions)
    return matches, frozenset(expressions)


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
    model_match_memo: _ModelMatchMemo | None = None,
) -> list[dict[str, typing.Any]]:
    details: list[dict[str, typing.Any]] = []
    serialized_items = tuple(_walk_serialized(value, path))
    safe_models, safe_parameter_expressions = _safe_model_matches(
        value,
        model_reference,
        path,
        model_match_memo=model_match_memo,
        serialized_items=serialized_items,
    )
    for item_path, item, malformed in serialized_items:
        if any(
            item_path == safe_path or item_path.startswith(f"{safe_path}/")
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
        if item is None:
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
    model_match_memo: _ModelMatchMemo | None = None,
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
        model_match_memo=model_match_memo,
    )


def lmfit_model_safe_parameter_expressions(
    serialized_model: str,
    model_reference: str,
) -> dict[str, str]:
    """Return model hints only for an exact model generated by local library code."""
    try:
        value = json.loads(serialized_model)
    except ValueError:
        return {}
    matches, expressions = _safe_model_matches(value, model_reference)
    if len(matches) != 1:
        return {}
    return dict(expressions)


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
    model_match_memo: _ModelMatchMemo = {}
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
                            model_match_memo=model_match_memo,
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
