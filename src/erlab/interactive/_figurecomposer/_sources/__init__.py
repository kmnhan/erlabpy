"""Registered payload types for Figure Composer sources."""

from __future__ import annotations

import copy
import dataclasses
import typing
import urllib.parse

from erlab.interactive._figurecomposer._fit_sources import FigureFitSourceSnapshot

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    import xarray as xr

    from erlab.interactive._code_trust._core import CodeTrustEntry
    from erlab.interactive._figurecomposer._model._state import FigureOperationState


@dataclasses.dataclass(frozen=True)
class ArraySourcePayload:
    """Effective array and its optional unselected parent."""

    data: xr.DataArray | None
    selection_base: xr.DataArray | None = None


@dataclasses.dataclass(frozen=True)
class EncodedSourcePayload:
    """JSON metadata and separately stored native array slots."""

    metadata: dict[str, typing.Any]
    arrays: Mapping[str, xr.DataArray] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class OpaqueSourcePayload:
    """Preserved source data whose type is not registered in this runtime."""

    kind: str
    encoded: EncodedSourcePayload


@dataclasses.dataclass(frozen=True)
class SourceSpec[PayloadT]:
    """Storage and execution operations for one explicitly registered type.

    Decoding and safe value access must not execute stored Python objects.
    Callable reconstruction belongs in ``materialize`` and requires trust checks.
    """

    kind: str
    payload_type: type[PayloadT]
    duplicate: Callable[[PayloadT], PayloadT]
    encode: Callable[[PayloadT], EncodedSourcePayload]
    decode: Callable[[EncodedSourcePayload], PayloadT]
    safe_value: Callable[[PayloadT], object]
    label: str = ""
    supports_selection: bool = False
    is_available: Callable[[PayloadT], bool] = lambda payload: True
    describe: Callable[[PayloadT], tuple[str, ...]] = lambda payload: ()
    materialize: Callable[[PayloadT], object] | None = None
    trust_entries: Callable[[PayloadT, str, str], tuple[CodeTrustEntry, ...]] | None = (
        None
    )
    default_operation: Callable[[str], FigureOperationState] | None = None


_SOURCE_SPECS: dict[str, SourceSpec[object]] = {}


def register_source_spec(spec: SourceSpec[typing.Any]) -> None:
    """Register an in-process type without importing code named by saved files."""
    if not spec.kind or spec.kind in _SOURCE_SPECS:
        raise ValueError(f"Source type {spec.kind!r} is already registered or empty.")
    if any(
        existing.payload_type is spec.payload_type
        for existing in _SOURCE_SPECS.values()
    ):
        raise ValueError(f"Payload type {spec.payload_type!r} is already registered.")
    _SOURCE_SPECS[spec.kind] = spec


def source_spec_for_kind(kind: str) -> SourceSpec[object]:
    """Return a registered type specification."""
    return _SOURCE_SPECS[kind]


def source_spec_for_payload(payload: object) -> SourceSpec[object]:
    """Resolve a payload through explicit type registrations."""
    for spec in _SOURCE_SPECS.values():
        if type(payload) is spec.payload_type:
            return spec
    for spec in _SOURCE_SPECS.values():
        if isinstance(payload, spec.payload_type):
            return spec
    raise TypeError(f"Unregistered Figure Composer payload type: {type(payload)!r}.")


def _is_array_source_payload(
    payload: object,
) -> typing.TypeGuard[ArraySourcePayload]:
    """Identify payloads that use the built-in array source handling."""
    return (
        isinstance(payload, ArraySourcePayload)
        and source_spec_for_payload(payload).kind == "array"
    )


def source_payload_kind(payload: object) -> str:
    return (
        payload.kind
        if isinstance(payload, OpaqueSourcePayload)
        else source_spec_for_payload(payload).kind
    )


def duplicate_source_payload(payload: object) -> object:
    if isinstance(payload, OpaqueSourcePayload):
        return OpaqueSourcePayload(
            payload.kind,
            EncodedSourcePayload(
                copy.deepcopy(payload.encoded.metadata),
                {
                    name: data.copy(deep=False)
                    for name, data in payload.encoded.arrays.items()
                },
            ),
        )
    return source_spec_for_payload(payload).duplicate(payload)


def encode_source_payload(payload: object) -> EncodedSourcePayload:
    """Encode one source without materializing executable content."""
    if isinstance(payload, OpaqueSourcePayload):
        return payload.encoded
    return source_spec_for_payload(payload).encode(payload)


def decode_source_payload(kind: str, payload: EncodedSourcePayload) -> object:
    """Decode registered storage, retaining unsupported types as opaque data."""
    spec = _SOURCE_SPECS.get(kind)
    return OpaqueSourcePayload(kind, payload) if spec is None else spec.decode(payload)


def _duplicate_array(payload: ArraySourcePayload) -> ArraySourcePayload:
    return ArraySourcePayload(
        None if payload.data is None else payload.data.copy(deep=False),
        None
        if payload.selection_base is None
        else payload.selection_base.copy(deep=False),
    )


def _encode_array(payload: ArraySourcePayload) -> EncodedSourcePayload:
    return EncodedSourcePayload(
        {},
        {
            name: data
            for name, data in (
                ("data", payload.data),
                ("selection_base", payload.selection_base),
            )
            if data is not None
        },
    )


def _array_value(payload: ArraySourcePayload) -> xr.DataArray | None:
    return payload.data


def _fit_trust_entries(
    payload: FigureFitSourceSnapshot, alias: str, location_prefix: str
) -> tuple[CodeTrustEntry, ...]:
    from erlab.interactive._fit_code_trust import lmfit_result_code_entry

    entry = lmfit_result_code_entry(
        payload.blob,
        feature="erlab.figure-composer.fit-result",
        location=(
            f"{location_prefix}/sources/{urllib.parse.quote(alias, safe='')}/fit-result"
        ),
    )
    return () if entry is None else (entry,)


def _fit_operation(alias: str) -> FigureOperationState:
    from erlab.interactive._figurecomposer._model._state import FigureOperationState

    return FigureOperationState.fit_plot(source=alias)


register_source_spec(
    SourceSpec(
        kind="array",
        label="Array",
        supports_selection=True,
        is_available=lambda payload: payload.data is not None,
        payload_type=ArraySourcePayload,
        duplicate=_duplicate_array,
        encode=_encode_array,
        decode=lambda payload: ArraySourcePayload(
            payload.arrays.get("data"), payload.arrays.get("selection_base")
        ),
        safe_value=_array_value,
    )
)
register_source_spec(
    SourceSpec(
        kind="fit",
        label="Fit result",
        describe=lambda payload: tuple(
            value
            for value in (
                payload.model_name,
                ", ".join(payload.independent_vars),
            )
            if value
        ),
        payload_type=FigureFitSourceSnapshot,
        duplicate=lambda payload: payload,
        encode=lambda payload: EncodedSourcePayload(payload.to_payload()),
        decode=lambda payload: FigureFitSourceSnapshot.from_payload(payload.metadata),
        safe_value=lambda payload: payload.native_dataset,
        materialize=lambda payload: payload.result_dataset(),
        trust_entries=_fit_trust_entries,
        default_operation=_fit_operation,
    )
)
