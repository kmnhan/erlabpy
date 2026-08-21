"""Metadata for opaque saved payloads that can execute Python when decoded."""

from __future__ import annotations

import hashlib
import json
import typing

from erlab.interactive._code_trust._core import CodeTrustEntry

if typing.TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, MutableMapping

CODE_PAYLOAD_ENTRIES_ATTR = "erlab_code_trust_payload_entries"
_PAYLOAD_SHA256_KEY = "payload_sha256"


def create_payload_entry(
    feature: str,
    location: str,
    code: str,
    payload: bytes,
    context: Mapping[str, typing.Any] | None = None,
) -> CodeTrustEntry:
    """Create an entry that binds an opaque executable payload by SHA-256."""
    entry_context = {} if context is None else dict(context)
    if _PAYLOAD_SHA256_KEY in entry_context:
        raise ValueError(f"{_PAYLOAD_SHA256_KEY!r} is reserved by code trust")
    entry_context[_PAYLOAD_SHA256_KEY] = hashlib.sha256(payload).hexdigest()
    return CodeTrustEntry(feature, location, code, entry_context)


def _validated_entries(entries: Iterable[CodeTrustEntry]) -> tuple[CodeTrustEntry, ...]:
    validated = tuple(entries)
    locations: set[tuple[str, str]] = set()
    for entry in validated:
        digest = entry.context.get(_PAYLOAD_SHA256_KEY)
        try:
            valid_digest = (
                isinstance(digest, str)
                and len(digest) == 64
                and len(bytes.fromhex(digest)) == 32
            )
        except ValueError:
            valid_digest = False
        if not valid_digest:
            raise TypeError("Code payload entry contains an invalid SHA-256 digest")
        key = (entry.feature, entry.location)
        if key in locations:
            raise ValueError("Code payload entries must have unique locations")
        locations.add(key)
    return validated


def code_payload_entries_from_metadata(
    attrs: Mapping[str, typing.Any],
) -> tuple[CodeTrustEntry, ...]:
    """Read validated opaque-payload entries from saved document metadata."""
    raw = attrs.get(CODE_PAYLOAD_ENTRIES_ATTR)
    if raw is None:
        return ()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if not isinstance(raw, str):
        raise TypeError("Saved code payload metadata must be JSON text")
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise TypeError("Saved code payload metadata is invalid JSON") from exc
    if not isinstance(payload, list):
        raise TypeError("Saved code payload metadata must be a JSON list")

    entries: list[CodeTrustEntry] = []
    for item in payload:
        if not isinstance(item, dict) or set(item) != {
            "code",
            "context",
            "feature",
            "location",
        }:
            raise TypeError("Saved code payload entry has invalid fields")
        context = item["context"]
        if not isinstance(context, dict):
            raise TypeError("Saved code payload entry context must be an object")
        entries.append(
            CodeTrustEntry(item["feature"], item["location"], item["code"], context)
        )
    return _validated_entries(entries)


def store_code_payload_entries(
    attrs: MutableMapping[typing.Any, typing.Any],
    entries: Iterable[CodeTrustEntry],
) -> None:
    """Store opaque-payload entries as deterministic document metadata."""
    validated = _validated_entries(entries)
    if not validated:
        attrs.pop(CODE_PAYLOAD_ENTRIES_ATTR, None)
        return
    attrs[CODE_PAYLOAD_ENTRIES_ATTR] = json.dumps(
        [entry.payload() for entry in validated],
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
