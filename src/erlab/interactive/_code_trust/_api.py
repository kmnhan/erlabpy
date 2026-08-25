"""Small functional interface for executable-document code trust."""

from __future__ import annotations

import dataclasses
import json
import typing

from erlab.interactive._code_trust._core import (
    CodeTrustEntry,
    CodeTrustEntrySource,
    CodeTrustManifest,
    CodeTrustReason,
)

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    _T = typing.TypeVar("_T")


def create_entry(
    feature: str,
    location: str,
    code: str,
    context: Mapping[str, typing.Any] | None = None,
) -> CodeTrustEntry:
    """Create one validated executable-content entry."""
    return CodeTrustEntry(feature, location, code, {} if context is None else context)


def create_manifest(
    domain: str,
    policy_version: int,
    entries: Iterable[CodeTrustEntry],
) -> CodeTrustManifest:
    """Create one validated executable-content manifest."""
    return CodeTrustManifest(domain, policy_version, tuple(entries))


def manifest_has_code(manifest: CodeTrustManifest) -> bool:
    """Return whether a manifest contains executable content."""
    return bool(manifest.entries)


def relocate_manifest_entries(
    manifest: CodeTrustManifest,
    *,
    location_prefix: str,
    remove_location_prefix: str = "",
) -> tuple[CodeTrustEntry, ...]:
    """Return entries with locations relative to an embedding document."""
    return tuple(
        create_entry(
            entry.feature,
            (
                f"{location_prefix}/"
                f"{entry.location.removeprefix(remove_location_prefix)}"
            ),
            entry.code,
            entry.context,
        )
        for entry in manifest.entries
    )


def relocate_document_trust(
    trust: _DocumentTrust,
    manifest: CodeTrustManifest,
    *,
    location_prefix: str,
) -> _DocumentTrust:
    """Move feature-relative trust into an embedding document location."""
    relocated = create_manifest(
        manifest.domain,
        manifest.policy_version,
        relocate_manifest_entries(manifest, location_prefix=location_prefix),
    )
    if (
        trust.reason == CodeTrustReason.SIGNATURE
        and trust.manifest_identity != manifest.canonical_bytes()
    ):
        return _document_trust(CodeTrustReason.UNTRUSTED, relocated)
    if document_trust_has_trusted_lineage(trust):
        return _document_trust(CodeTrustReason.LOCAL_LINEAGE, relocated)
    if trust.reason == CodeTrustReason.NO_EXECUTABLE_CODE:
        return _document_trust(CodeTrustReason.NO_EXECUTABLE_CODE, relocated)
    local_document_identities = {
        relocated_entry.document_identity()
        for source_entry, relocated_entry in zip(
            manifest.entries, relocated.entries, strict=True
        )
        if source_entry.document_identity() in trust.local_document_identities
    }
    return _document_trust(
        CodeTrustReason.UNTRUSTED,
        relocated,
        local_document_identities=local_document_identities,
    )


def _group_manifest_review_entries(
    manifest: CodeTrustManifest,
) -> tuple[tuple[CodeTrustEntry, tuple[str, ...]], ...]:
    """Group equal execution content while retaining its document locations."""
    representatives: dict[bytes, CodeTrustEntry] = {}
    locations: dict[bytes, list[str]] = {}
    for entry in manifest.entries:
        identity = entry.execution_identity()
        representatives.setdefault(identity, entry)
        locations.setdefault(identity, []).append(entry.location)
    return tuple(
        (entry, tuple(locations[identity]))
        for identity, entry in representatives.items()
    )


def manifest_review_text(manifest: CodeTrustManifest) -> str:
    """Return compact plain text for review of executable manifest entries."""
    blocks: list[str] = []
    for entry, locations in _group_manifest_review_entries(manifest):
        location_label = (
            locations[0] if len(locations) == 1 else f"{len(locations)} locations"
        )
        block = f"[{entry.feature} — {location_label}]\n{entry.code}"
        if entry.context:
            context = json.dumps(
                entry.payload()["context"], ensure_ascii=False, sort_keys=True, indent=2
            )
            block = f"{block}\nExecution context:\n{context}"
        if len(locations) > 1:
            shown_locations = locations[:10]
            location_text = "\n".join(f"- {location}" for location in shown_locations)
            remaining = len(locations) - len(shown_locations)
            if remaining:
                location_text = f"{location_text}\n- and {remaining} more"
            block = f"{block}\nLocations:\n{location_text}"
        blocks.append(block)
    return "\n\n".join(blocks)


@dataclasses.dataclass(frozen=True, slots=True)
class _DocumentTrust:
    """Complete private trust state for one executable document."""

    reason: CodeTrustReason
    manifest: CodeTrustManifest | None
    manifest_identity: bytes | None
    execution_identities: frozenset[bytes]
    document_identities: frozenset[bytes]
    local_document_identities: frozenset[bytes]


@dataclasses.dataclass(frozen=True, slots=True)
class _ExecutionCapability:
    """Opaque allow-list of exact executable-content identities."""

    execution_identities: frozenset[bytes]


def _document_trust(
    reason: CodeTrustReason,
    manifest: CodeTrustManifest | None,
    *,
    local_document_identities: Iterable[bytes] = (),
) -> _DocumentTrust:
    execution_identities = (
        frozenset()
        if manifest is None
        else frozenset(entry.execution_identity() for entry in manifest.entries)
    )
    return _DocumentTrust(
        reason,
        manifest,
        None if manifest is None else manifest.canonical_bytes(),
        execution_identities,
        (
            frozenset()
            if manifest is None
            else frozenset(entry.document_identity() for entry in manifest.entries)
        ),
        frozenset(local_document_identities),
    )


def new_document_trust() -> _DocumentTrust:
    """Create trust state for a new local document."""
    return _document_trust(CodeTrustReason.LOCAL_LINEAGE, None)


def external_document_trust(
    manifest: CodeTrustManifest | None,
) -> _DocumentTrust:
    """Create safe trust state for externally supplied document content."""
    has_executable_code = manifest is not None and manifest_has_code(manifest)
    return _document_trust(
        CodeTrustReason.UNTRUSTED
        if has_executable_code
        else CodeTrustReason.NO_EXECUTABLE_CODE,
        manifest,
    )


def untrusted_document_trust(
    manifest: CodeTrustManifest | None = None,
) -> _DocumentTrust:
    """Create trust state for external content that must be treated as executable."""
    return _document_trust(CodeTrustReason.UNTRUSTED, manifest)


def trusted_location_document_trust() -> _DocumentTrust:
    """Create trust state for content from a trusted document location."""
    return _document_trust(CodeTrustReason.TRUSTED_LOCATION, None)


def merge_document_trust(
    current: _DocumentTrust,
    incoming: _DocumentTrust,
    *,
    replace: bool,
) -> _DocumentTrust:
    """Return document trust after a load or import operation."""
    if replace:
        return incoming
    if incoming == current:
        return current
    if incoming.reason == CodeTrustReason.NO_EXECUTABLE_CODE:
        return current
    reason = CodeTrustReason.LOCAL_LINEAGE
    if not (
        document_trust_has_trusted_lineage(current)
        and document_trust_has_trusted_lineage(incoming)
    ):
        reason = CodeTrustReason.UNTRUSTED
    local_document_identities: set[bytes] = set()
    if reason == CodeTrustReason.UNTRUSTED:
        local_document_identities.update(current.local_document_identities)
        local_document_identities.update(incoming.local_document_identities)
        if document_trust_has_trusted_lineage(current) and current.manifest is not None:
            local_document_identities.update(current.document_identities)
        if (
            document_trust_has_trusted_lineage(incoming)
            and incoming.manifest is not None
        ):
            local_document_identities.update(incoming.document_identities)
    return _document_trust(
        reason,
        None,
        local_document_identities=local_document_identities,
    )


def approve_document_trust(
    current: _DocumentTrust,
    manifest: CodeTrustManifest | None = None,
) -> _DocumentTrust:
    """Return locally approved trust while retaining the review manifest."""
    if manifest is None:
        manifest = current.manifest
    return _document_trust(CodeTrustReason.LOCAL_LINEAGE, manifest)


def bind_document_trust_manifest(
    trust: _DocumentTrust,
    manifest: CodeTrustManifest,
) -> _DocumentTrust:
    """Bind trust to the last committed executable manifest.

    This binding is process-local state. It does not create a durable signature. A
    signature remains valid only for its complete original manifest.
    """
    current_identities = {entry.document_identity() for entry in manifest.entries}
    local_document_identities = trust.local_document_identities & current_identities
    reason = trust.reason
    if (
        reason == CodeTrustReason.SIGNATURE
        and trust.manifest_identity != manifest.canonical_bytes()
    ):
        reason = CodeTrustReason.UNTRUSTED
    return _document_trust(
        reason,
        manifest,
        local_document_identities=local_document_identities,
    )


def _document_trust_after_save(
    current: _DocumentTrust,
    manifest: CodeTrustManifest,
    *,
    saved_trusted_lineage: bool,
    signature_stored: bool,
) -> _DocumentTrust:
    if not manifest_has_code(manifest):
        return _document_trust(CodeTrustReason.LOCAL_LINEAGE, manifest)
    if not saved_trusted_lineage:
        if document_trust_has_trusted_lineage(current):
            # Approval can occur while an untrusted snapshot is being saved. The
            # completed save must not overwrite that newer document decision.
            return current
        committed_identities = {entry.document_identity() for entry in manifest.entries}
        return _document_trust(
            CodeTrustReason.UNTRUSTED,
            manifest,
            local_document_identities=(
                current.local_document_identities & committed_identities
            ),
        )
    if not document_trust_has_trusted_lineage(current):
        # A payload check can revoke trust while an approved snapshot is being
        # saved. The completed save must not restore the older decision.
        return current
    return _document_trust(
        CodeTrustReason.SIGNATURE
        if signature_stored
        else CodeTrustReason.LOCAL_LINEAGE,
        manifest,
    )


def authorize_document_execution(
    trust: _DocumentTrust,
    entries: CodeTrustEntrySource,
) -> tuple[_DocumentTrust, bool]:
    """Apply the document policy and return updated trust plus authorization."""
    if document_trust_has_trusted_lineage(trust) and (
        trust.reason != CodeTrustReason.SIGNATURE
    ):
        return trust, True
    resolved_entries = tuple(entries() if callable(entries) else entries)
    if not resolved_entries:
        return trust, True
    if trust.reason == CodeTrustReason.SIGNATURE and all(
        entry.execution_identity() in trust.execution_identities
        for entry in resolved_entries
    ):
        return trust, True
    if trust.reason == CodeTrustReason.UNTRUSTED and all(
        entry.document_identity() in trust.local_document_identities
        for entry in resolved_entries
    ):
        return trust, True
    return (
        trust
        if trust.reason == CodeTrustReason.UNTRUSTED
        else untrusted_document_trust(trust.manifest),
        False,
    )


def issue_execution_capability(
    trust: _DocumentTrust,
    entries: CodeTrustEntrySource,
) -> tuple[_DocumentTrust, object | None]:
    """Issue an allow-list for the authorized subset of exact entry content."""
    resolved_entries = tuple(entries() if callable(entries) else entries)
    trust, allowed = authorize_document_execution(trust, resolved_entries)
    allowed_identities = frozenset(
        entry.execution_identity() for entry in resolved_entries
    )
    if not allowed and trust.reason == CodeTrustReason.UNTRUSTED:
        locally_allowed = {
            entry.execution_identity()
            for entry in resolved_entries
            if entry.document_identity() in trust.local_document_identities
        }
        externally_owned = {
            entry.execution_identity()
            for entry in resolved_entries
            if entry.document_identity() not in trust.local_document_identities
        }
        # A location-free capability cannot distinguish equal content at two
        # locations. Do not issue that identity when this request contains both.
        allowed_identities = frozenset(locally_allowed - externally_owned)
    capability = (
        _ExecutionCapability(allowed_identities)
        if allowed or allowed_identities
        else None
    )
    return trust, capability


def issue_complete_execution_capability(
    trust: _DocumentTrust,
    entries: CodeTrustEntrySource,
) -> tuple[_DocumentTrust, object | None]:
    """Issue a capability only when it allows the complete request."""
    resolved_entries = tuple(entries() if callable(entries) else entries)
    trust, capability = issue_execution_capability(trust, resolved_entries)
    if not execution_capability_allows(capability, resolved_entries):
        capability = None
    return trust, capability


def _execution_identities_allowed_before_local_edit(
    trust: _DocumentTrust,
    entries: Iterable[CodeTrustEntry],
) -> frozenset[bytes]:
    resolved_entries = tuple(entries)
    if trust.reason == CodeTrustReason.SIGNATURE:
        return frozenset(
            entry.execution_identity()
            for entry in resolved_entries
            if entry.execution_identity() in trust.execution_identities
        )
    if trust.reason == CodeTrustReason.UNTRUSTED:
        return frozenset(
            entry.execution_identity()
            for entry in resolved_entries
            if entry.document_identity() in trust.local_document_identities
        )
    if trust.reason in {
        CodeTrustReason.LOCAL_LINEAGE,
        CodeTrustReason.TRUSTED_LOCATION,
    }:
        return frozenset(entry.execution_identity() for entry in resolved_entries)
    return frozenset()


def issue_local_edit_capability(
    trust: _DocumentTrust,
    execution_entries: CodeTrustEntrySource,
    *,
    edited_entries: CodeTrustEntrySource,
) -> tuple[_DocumentTrust, object | None]:
    """Authorize validation of an explicit local executable-content edit.

    Entries supplied in ``edited_entries`` are locally authored. The capability
    allows those entries plus each requested entry allowed by the prior stored-content
    policy. It can therefore guard separate entry boundaries in one mixed graph. The
    returned trust value is prospective: the document host must apply it only after
    validation and commit both succeed.
    """
    resolved_entries = tuple(
        execution_entries() if callable(execution_entries) else execution_entries
    )
    resolved_edited_entries = tuple(
        edited_entries() if callable(edited_entries) else edited_entries
    )
    prior_allowed_identities = _execution_identities_allowed_before_local_edit(
        trust, resolved_entries
    )
    edited_document_identities = frozenset(
        entry.document_identity() for entry in resolved_edited_entries
    )
    edited_execution_identities = {
        entry.execution_identity()
        for entry in resolved_entries
        if entry.document_identity() in edited_document_identities
    }
    if trust.reason == CodeTrustReason.UNTRUSTED:
        externally_owned_identities = {
            entry.execution_identity()
            for entry in resolved_entries
            if entry.document_identity()
            not in (trust.local_document_identities | edited_document_identities)
        }
        # A location-free capability cannot distinguish a local edit from equal
        # external content at another location. Prior-policy authorization remains
        # valid because it predates this edit request.
        edited_execution_identities.difference_update(externally_owned_identities)
    allowed_identities = frozenset(
        prior_allowed_identities | edited_execution_identities
    )

    prospective_trust = trust
    if trust.reason in {
        CodeTrustReason.NO_EXECUTABLE_CODE,
        CodeTrustReason.SIGNATURE,
        CodeTrustReason.UNTRUSTED,
    }:
        prospective_trust = _document_trust(
            CodeTrustReason.LOCAL_LINEAGE,
            trust.manifest,
        )
    return prospective_trust, _ExecutionCapability(allowed_identities)


def commit_local_edit_trust(
    trust: _DocumentTrust,
    capability: object | None,
    previous_document_entries: CodeTrustEntrySource,
    document_entries: CodeTrustEntrySource,
    *,
    edited_entries: CodeTrustEntrySource,
    document_manifest: CodeTrustManifest | None = None,
) -> _DocumentTrust:
    """Commit local lineage only when the full document remains authorized.

    The validation capability is intentionally scoped to one feature edit. The
    document host must call this function after the edit commits with the complete
    executable inventories from before and after the transaction. New entries created
    during validation belong to the local edit. Retained entries must still pass the
    prior document policy. This prevents one local edit from authorizing unrelated
    external code.
    """
    resolved_previous_entries = tuple(
        previous_document_entries()
        if callable(previous_document_entries)
        else previous_document_entries
    )
    resolved_document_entries = tuple(
        document_entries() if callable(document_entries) else document_entries
    )
    resolved_edited_entries = tuple(
        edited_entries() if callable(edited_entries) else edited_entries
    )
    if not isinstance(capability, _ExecutionCapability) or not all(
        entry.execution_identity() in capability.execution_identities
        for entry in resolved_edited_entries
    ):
        return trust
    previous_execution_identities = frozenset(
        entry.execution_identity() for entry in resolved_previous_entries
    )
    current_identities = frozenset(
        entry.document_identity() for entry in resolved_document_entries
    )
    local_identities = set(trust.local_document_identities)
    local_identities.update(
        entry.document_identity()
        for entry in resolved_edited_entries
        if entry.document_identity() in current_identities
    )
    # Validation can create derived entries. Adopt only content that is new by
    # location-free execution identity. A location-only move of external content
    # must not become a local edit.
    local_identities.update(
        entry.document_identity()
        for entry in resolved_document_entries
        if entry.execution_identity() not in previous_execution_identities
    )
    local_identities.intersection_update(current_identities)

    if trust.reason == CodeTrustReason.SIGNATURE:
        prior_policy_identities = trust.document_identities
    elif trust.reason == CodeTrustReason.UNTRUSTED:
        prior_policy_identities = trust.local_document_identities
    elif trust.reason in {
        CodeTrustReason.LOCAL_LINEAGE,
        CodeTrustReason.TRUSTED_LOCATION,
    }:
        committed = trust
    else:
        prior_policy_identities = frozenset()
    if trust.reason not in {
        CodeTrustReason.LOCAL_LINEAGE,
        CodeTrustReason.TRUSTED_LOCATION,
    }:
        if current_identities <= prior_policy_identities | local_identities:
            committed = _document_trust(CodeTrustReason.LOCAL_LINEAGE, trust.manifest)
        else:
            committed = _document_trust(
                CodeTrustReason.UNTRUSTED,
                trust.manifest,
                local_document_identities=local_identities,
            )
    if document_manifest is not None:
        committed = bind_document_trust_manifest(committed, document_manifest)
    return committed


def execution_capability_allows(
    capability: object | None,
    entries: CodeTrustEntrySource,
) -> bool:
    """Return whether a capability allows all supplied exact content identities."""
    resolved_entries = tuple(entries() if callable(entries) else entries)
    if not resolved_entries:
        return True
    if not isinstance(capability, _ExecutionCapability):
        return False
    return all(
        entry.execution_identity() in capability.execution_identities
        for entry in resolved_entries
    )


def execute_with_capability(
    capability: object | None,
    entries: CodeTrustEntrySource,
    execute: Callable[[], _T],
) -> tuple[bool, _T | None]:
    """Execute one callback only when the capability allows its exact entries."""
    resolved_entries = tuple(entries() if callable(entries) else entries)
    if not execution_capability_allows(capability, resolved_entries):
        return False, None
    return True, execute()


def verify_document_payload_entries(
    trust: _DocumentTrust,
    expected_entries: Iterable[CodeTrustEntry],
    actual_entries: Iterable[CodeTrustEntry],
) -> _DocumentTrust:
    """Verify opaque payload bytes before trusted lineage can be preserved.

    A legacy external document can contain payloads without digest metadata. Such a
    document uses the normal approval flow after its actual payload inventory is known.
    A document that was already authorized must declare every payload unless it is in a
    trusted location. Once digest metadata exists, it must match the bytes exactly.
    """
    expected = tuple(expected_entries)
    actual = tuple(actual_entries)
    if expected == actual:
        return trust
    if expected or (
        document_trust_has_trusted_lineage(trust)
        and trust.reason != CodeTrustReason.TRUSTED_LOCATION
    ):
        return untrusted_document_trust(trust.manifest)
    return trust


def document_trust_is_trusted(trust: _DocumentTrust) -> bool:
    """Return whether document executable content is authorized."""
    return trust.reason != CodeTrustReason.UNTRUSTED


def document_trust_has_trusted_lineage(trust: _DocumentTrust) -> bool:
    """Return whether later document saves can preserve durable trust."""
    return trust.reason in {
        CodeTrustReason.SIGNATURE,
        CodeTrustReason.TRUSTED_LOCATION,
        CodeTrustReason.LOCAL_LINEAGE,
    }


def document_trust_needs_review(trust: _DocumentTrust) -> bool:
    """Return whether the document needs approval before code can run."""
    return trust.reason == CodeTrustReason.UNTRUSTED


def document_trust_description(trust: _DocumentTrust) -> str:
    """Return the user-facing description of the document trust state."""
    if trust.manifest is not None and not manifest_has_code(trust.manifest):
        return "No stored executable content"
    return {
        CodeTrustReason.NO_EXECUTABLE_CODE: "No stored executable content",
        CodeTrustReason.SIGNATURE: "Trusted by saved signature",
        CodeTrustReason.TRUSTED_LOCATION: "Trusted location",
        CodeTrustReason.LOCAL_LINEAGE: "Trusted local document",
        CodeTrustReason.UNTRUSTED: "Stored executable content is paused",
    }[trust.reason]
