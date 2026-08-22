"""Validated values and canonical JSON for code trust."""

from __future__ import annotations

import dataclasses
import enum
import json
import math
import typing
from collections.abc import Callable, Iterable, Mapping

JSONScalar = str | int | float | bool | None
JSONValue = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
CodeTrustEntrySource = (
    Iterable["CodeTrustEntry"] | Callable[[], Iterable["CodeTrustEntry"]]
)


class CodeTrustReason(enum.StrEnum):
    """Reason for one executable-document trust decision."""

    NO_EXECUTABLE_CODE = "no_executable_code"
    SIGNATURE = "signature"
    TRUSTED_LOCATION = "trusted_location"
    LOCAL_LINEAGE = "local_lineage"
    UNTRUSTED = "untrusted"


def _normalize_json(value: typing.Any, path: str) -> JSONValue:
    if value is None or isinstance(value, str | bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, JSONValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string mapping key")
            normalized[key] = _normalize_json(item, f"{path}.{key}")
        return normalized
    if isinstance(value, list):
        return [
            _normalize_json(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(f"{path} contains unsupported value {type(value).__name__}")


def _canonical_json(value: JSONValue) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


@dataclasses.dataclass(frozen=True, slots=True)
class CodeTrustEntry:
    """One executable contribution to a document trust manifest."""

    feature: str
    location: str
    code: str
    context: Mapping[str, typing.Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.feature, str):
            raise TypeError("Code trust feature must be a string")
        if not self.feature.strip():
            raise ValueError("Code trust feature must not be empty")
        if not isinstance(self.location, str):
            raise TypeError("Code trust location must be a string")
        if not self.location.strip():
            raise ValueError("Code trust location must not be empty")
        if not isinstance(self.code, str):
            raise TypeError("Code trust code must be a string")
        if not isinstance(self.context, Mapping):
            raise TypeError("Code trust context must be a mapping")
        context = _normalize_json(self.context, "context")
        # Keep the signed value independent from mutable feature state. The
        # normalization recursively copies all supported mappings and lists.
        object.__setattr__(
            self, "context", typing.cast("dict[str, JSONValue]", context)
        )

    def payload(self) -> dict[str, JSONValue]:
        """Return the signed JSON payload for this entry."""
        return {
            "code": self.code,
            "context": typing.cast(
                "dict[str, JSONValue]", _normalize_json(self.context, "context")
            ),
            "feature": self.feature,
            "location": self.location,
        }

    def execution_identity(self) -> bytes:
        """Return canonical content identity without its document location."""
        payload = self.payload()
        payload.pop("location")
        return _canonical_json(payload)

    def document_identity(self) -> bytes:
        """Return canonical content identity at its exact document location."""
        return _canonical_json(self.payload())


@dataclasses.dataclass(frozen=True, slots=True)
class CodeTrustManifest:
    """Ordered executable content signed as one trust unit."""

    domain: str
    policy_version: int
    entries: tuple[CodeTrustEntry, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.domain, str):
            raise TypeError("Code trust domain must be a string")
        if not self.domain.strip():
            raise ValueError("Code trust domain must not be empty")
        if type(self.policy_version) is not int:
            raise TypeError("Code trust policy version must be an integer")
        if self.policy_version < 1:
            raise ValueError("Code trust policy version must be positive")
        if not isinstance(self.entries, tuple):
            raise TypeError("Code trust manifest entries must be a tuple")
        if not all(isinstance(entry, CodeTrustEntry) for entry in self.entries):
            raise TypeError("Code trust manifest entries must be CodeTrustEntry values")

    @property
    def has_executable_code(self) -> bool:
        """Return whether this manifest contains executable entries."""
        return bool(self.entries)

    def payload(self) -> dict[str, JSONValue]:
        """Return the complete signed JSON payload."""
        return {
            "domain": self.domain,
            "entries": [entry.payload() for entry in self.entries],
            "policy_version": self.policy_version,
        }

    def canonical_bytes(self) -> bytes:
        """Return a deterministic byte representation for signing."""
        return _canonical_json(self.payload())
