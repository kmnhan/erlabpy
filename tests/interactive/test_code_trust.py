from __future__ import annotations

import json
import os
import pathlib
import sqlite3
import typing
from contextlib import closing

import pytest
from qtpy import QtCore, QtWidgets

import erlab.interactive._code_trust as code_trust
import erlab.interactive._code_trust._application as _application
from erlab.interactive._code_trust import (
    approve_document_trust,
    document_trust_has_trusted_lineage,
    document_trust_is_trusted,
    external_document_trust,
    merge_document_trust,
    new_document_trust,
    trusted_location_document_trust,
    untrusted_document_trust,
)
from erlab.interactive._code_trust._api import _document_trust_after_save
from erlab.interactive._code_trust._core import CodeTrustEntry, CodeTrustManifest
from erlab.interactive._code_trust._locations import (
    document_path_is_trusted,
    validate_trusted_location,
)
from erlab.interactive._code_trust._notary import CodeTrustError, CodeTrustNotary
from erlab.interactive._code_trust._payloads import (
    code_payload_entries_from_metadata,
    store_code_payload_entries,
)
from erlab.interactive._code_trust._ui import confirm_code_trust


def _entry(
    code: str = "value = 1",
    *,
    feature: str = "test.code",
    location: str = "window/0",
    context: dict[str, object] | None = None,
) -> CodeTrustEntry:
    return CodeTrustEntry(
        feature,
        location,
        code,
        {"enabled": True, "order": [0, 1]} if context is None else context,
    )


def _manifest(
    code: str = "value = 1", *, domain: str = "erlab.workspace", version: int = 1
) -> CodeTrustManifest:
    return CodeTrustManifest(domain, version, (_entry(code),))


def _signed_trust(manifest: CodeTrustManifest | None = None):
    return _document_trust_after_save(
        new_document_trust(),
        _manifest() if manifest is None else manifest,
        saved_trusted_lineage=True,
        signature_stored=True,
    )


def _trust_after_save(
    trust,
    manifest: CodeTrustManifest,
    *,
    trusted_lineage: bool,
):
    return _document_trust_after_save(
        trust,
        manifest,
        saved_trusted_lineage=trusted_lineage,
        signature_stored=True,
    )


def test_code_trust_package_exposes_only_the_functional_facade() -> None:
    for implementation_type in (
        "CodeTrustDecision",
        "CodeTrustEntry",
        "CodeTrustManifest",
        "CodeTrustNotary",
        "CodeTrustReason",
        "_DocumentTrust",
        "_ExecutionCapability",
    ):
        assert not hasattr(code_trust, implementation_type)
    for obsolete_name in (
        "document_trust_is_local_approval",
        "document_trust_is_signed",
        "mark_document_trust_edited",
        "register_code_reference",
        "register_code_reference_loader",
        "resolve_code_reference",
    ):
        assert not hasattr(code_trust, obsolete_name)


def test_code_trust_manifest_canonicalization_and_validation() -> None:
    first = _manifest()
    second = CodeTrustManifest(
        "erlab.workspace",
        1,
        (_entry(context={"order": [0, 1], "enabled": True}),),
    )

    assert first.canonical_bytes() == second.canonical_bytes()
    with pytest.raises(ValueError, match="non-finite"):
        CodeTrustEntry("test", "here", "code", {"value": float("nan")})
    with pytest.raises(TypeError, match="non-string"):
        CodeTrustEntry("test", "here", "code", {1: "bad"})
    with pytest.raises(TypeError, match="unsupported"):
        CodeTrustEntry("test", "here", "code", {"values": (1, 2)})
    with pytest.raises(ValueError, match="positive"):
        CodeTrustManifest("test", 0, ())


def test_code_trust_entry_snapshots_mutable_context() -> None:
    context = {"enabled": True, "inputs": [{"name": "data"}]}
    entry = code_trust.create_entry("test", "operation/0", "run()", context)
    original = entry.payload()

    context["enabled"] = False
    context["inputs"][0]["name"] = "changed"

    assert entry.payload() == original


def test_manifest_review_includes_signed_execution_context() -> None:
    manifest = _manifest()

    review = code_trust.manifest_review_text(manifest)
    serialized_context = json.dumps(
        manifest.entries[0].payload()["context"],
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    )

    assert manifest.entries[0].code in review
    assert serialized_context in review


@pytest.mark.parametrize("approve", [False, True])
def test_code_trust_review_dialog_returns_selected_action(
    qtbot, monkeypatch: pytest.MonkeyPatch, approve: bool
) -> None:
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)

    def select_action(message: QtWidgets.QMessageBox) -> None:
        if approve:
            button = next(
                candidate
                for candidate in message.buttons()
                if message.buttonRole(candidate)
                == QtWidgets.QMessageBox.ButtonRole.AcceptRole
            )
        else:
            button = message.button(QtWidgets.QMessageBox.StandardButton.Cancel)
            assert button is not None
        button.click()

    monkeypatch.setattr(QtWidgets.QMessageBox, "exec", select_action)

    assert (
        confirm_code_trust(
            parent,
            _manifest(),
            document_name="Document",
            object_name="test_code_trust_review_dialog",
            window_title="Review Stored Code",
        )
        is approve
    )


def test_relocated_entries_change_only_the_document_location() -> None:
    entry = _manifest().entries[0]

    relocated = code_trust.relocate_manifest_entries(
        CodeTrustManifest("test.first", 1, (entry,)),
        location_prefix="tools/0",
    )

    assert relocated[0].location == "tools/0/window/0"
    assert relocated[0].feature == entry.feature
    assert relocated[0].code == entry.code
    assert relocated[0].context == entry.context
    assert relocated[0].execution_identity() == entry.execution_identity()
    assert relocated[0].document_identity() != entry.document_identity()


def test_code_trust_lifecycle_is_centralized_behind_document_operations() -> None:
    local = new_document_trust()
    safe_external = external_document_trust(CodeTrustManifest("test", 1, ()))
    signed = _signed_trust()

    assert merge_document_trust(local, safe_external, replace=False) == local
    assert merge_document_trust(signed, signed, replace=False) == signed
    assert document_trust_is_trusted(safe_external)
    assert not document_trust_has_trusted_lineage(safe_external)
    assert not code_trust.authorize_document_execution(
        safe_external, _manifest().entries
    )[1]

    untrusted = untrusted_document_trust(_manifest())
    combined = merge_document_trust(local, untrusted, replace=False)
    assert not document_trust_is_trusted(combined)
    assert merge_document_trust(local, untrusted, replace=True) == untrusted

    combined_trusted = merge_document_trust(
        local,
        trusted_location_document_trust(),
        replace=False,
    )
    assert document_trust_is_trusted(combined_trusted)
    assert document_trust_has_trusted_lineage(combined_trusted)

    approved = approve_document_trust(untrusted)
    assert document_trust_is_trusted(approved)
    assert document_trust_has_trusted_lineage(approved)

    empty_manifest = CodeTrustManifest("test", 1, ())
    saved_empty_local = _trust_after_save(local, empty_manifest, trusted_lineage=True)
    assert document_trust_has_trusted_lineage(saved_empty_local)
    assert code_trust.authorize_document_execution(
        saved_empty_local, _manifest().entries
    )[1]
    saved_empty_external = _trust_after_save(
        untrusted, empty_manifest, trusted_lineage=False
    )
    assert document_trust_has_trusted_lineage(saved_empty_external)

    saved_untrusted = _trust_after_save(
        safe_external, _manifest(), trusted_lineage=False
    )
    assert code_trust.document_trust_needs_review(saved_untrusted)
    assert not document_trust_has_trusted_lineage(saved_untrusted)

    approved_during_save = _trust_after_save(
        approved, _manifest(), trusted_lineage=False
    )
    assert approved_during_save == approved

    revoked_during_save = _trust_after_save(
        untrusted, _manifest(), trusted_lineage=True
    )
    assert revoked_during_save == untrusted

    blocked, execution_allowed = code_trust.authorize_document_execution(
        safe_external,
        _manifest().entries,
    )
    assert not execution_allowed
    assert code_trust.document_trust_needs_review(blocked)


def test_binding_untrusted_manifest_updates_baseline_without_authorizing_external() -> (
    None
):
    external_entry = _entry(code="external()", location="external")
    local_entry = _entry(code="local()", location="local")
    original_manifest = CodeTrustManifest("test", 1, (external_entry,))
    current_manifest = CodeTrustManifest("test", 1, (external_entry, local_entry))
    untrusted = untrusted_document_trust(original_manifest)
    _, capability = code_trust.issue_local_edit_capability(
        untrusted,
        (local_entry,),
        edited_entries=(local_entry,),
    )
    mixed = code_trust.commit_local_edit_trust(
        untrusted,
        capability,
        original_manifest.entries,
        current_manifest.entries,
        edited_entries=(local_entry,),
    )

    bound = code_trust.bind_document_trust_manifest(mixed, current_manifest)

    assert bound.manifest == current_manifest
    assert code_trust.document_trust_needs_review(bound)
    assert code_trust.authorize_document_execution(bound, (local_entry,))[1]
    assert not code_trust.authorize_document_execution(bound, (external_entry,))[1]
    pruned = code_trust.bind_document_trust_manifest(bound, original_manifest)
    assert not code_trust.authorize_document_execution(pruned, (local_entry,))[1]


def test_binding_changed_signature_fails_closed() -> None:
    original_manifest = _manifest("signed()")
    changed_manifest = _manifest("changed()")
    signed = _signed_trust(original_manifest)

    assert code_trust.bind_document_trust_manifest(signed, original_manifest) == signed
    changed = code_trust.bind_document_trust_manifest(signed, changed_manifest)

    assert code_trust.document_trust_needs_review(changed)
    assert not code_trust.authorize_document_execution(
        changed, original_manifest.entries
    )[1]
    assert not code_trust.authorize_document_execution(
        changed, changed_manifest.entries
    )[1]

    original_manifest.entries[0].context["enabled"] = False
    mutated = code_trust.bind_document_trust_manifest(signed, original_manifest)
    assert code_trust.document_trust_needs_review(mutated)


def test_signed_document_authorizes_only_approved_execution_entries() -> None:
    manifest = _manifest()
    signed = _signed_trust(manifest)
    approved = manifest.entries[0]
    relocated = code_trust.create_entry(
        approved.feature,
        "embedded/tool/window/0",
        approved.code,
        approved.context,
    )

    unchanged, allowed = code_trust.authorize_document_execution(signed, (relocated,))

    assert allowed
    assert unchanged == signed
    for changed in (
        _entry(feature="test.other"),
        _entry(code="value = 2"),
        _entry(context={**approved.context, "enabled": False}),
    ):
        blocked, allowed = code_trust.authorize_document_execution(signed, (changed,))
        assert not allowed
        assert code_trust.document_trust_needs_review(blocked)


def test_execution_capability_binds_the_exact_authorized_content() -> None:
    manifest = _manifest()
    signed = _signed_trust(manifest)
    approved = manifest.entries[0]
    relocated = code_trust.create_entry(
        approved.feature,
        "embedded/window/0",
        approved.code,
        approved.context,
    )

    unchanged, capability = code_trust.issue_execution_capability(
        signed,
        (relocated,),
    )

    assert unchanged == signed
    assert capability is not None
    assert code_trust.execution_capability_allows(capability, (approved,))
    assert not code_trust.execution_capability_allows(True, (approved,))
    changed = _entry(code="value = 2")
    assert not code_trust.execution_capability_allows(capability, (changed,))

    local, local_capability = code_trust.issue_execution_capability(
        new_document_trust(),
        (approved,),
    )
    assert document_trust_has_trusted_lineage(local)
    assert local_capability is not None
    assert not code_trust.execution_capability_allows(local_capability, (changed,))

    blocked, blocked_capability = code_trust.issue_execution_capability(
        untrusted_document_trust(manifest),
        (approved,),
    )
    assert code_trust.document_trust_needs_review(blocked)
    assert blocked_capability is None


def test_execute_with_capability_keeps_the_check_at_the_execution_boundary() -> None:
    approved = _entry()
    changed = _entry(code="value = 2")
    _trust, capability = code_trust.issue_execution_capability(
        new_document_trust(),
        (approved,),
    )
    calls: list[str] = []

    executed, result = code_trust.execute_with_capability(
        capability,
        (approved,),
        lambda: calls.append("approved") or 1,
    )
    blocked, blocked_result = code_trust.execute_with_capability(
        capability,
        (changed,),
        lambda: calls.append("changed") or 2,
    )

    assert executed
    assert result == 1
    assert not blocked
    assert blocked_result is None
    assert calls == ["approved"]


def test_local_edit_capability_transition_matrix() -> None:
    edited_entry = _entry(code="value = 2")
    signed = _signed_trust()
    safe_external = external_document_trust(CodeTrustManifest("test", 1, ()))
    local = new_document_trust()
    trusted_location = trusted_location_document_trust()
    untrusted = untrusted_document_trust(_manifest())

    for trust, expected in (
        (signed, approve_document_trust(signed)),
        (safe_external, approve_document_trust(safe_external)),
        (local, local),
        (trusted_location, trusted_location),
        (untrusted, approve_document_trust(untrusted)),
    ):
        prospective, capability = code_trust.issue_local_edit_capability(
            trust,
            (edited_entry,),
            edited_entries=(edited_entry,),
        )

        assert prospective == expected
        assert capability is not None
        assert code_trust.execution_capability_allows(capability, (edited_entry,))


def test_local_edit_capability_applies_stored_policy_to_unchanged_entries() -> None:
    signed_entry = _entry()
    edited_entry = _entry(code="value = 2")
    unsigned_entry = _entry(code="external()")
    signed = _signed_trust(CodeTrustManifest("test", 1, (signed_entry,)))

    prospective, capability = code_trust.issue_local_edit_capability(
        signed,
        (signed_entry, edited_entry),
        edited_entries=(edited_entry,),
    )

    assert prospective == approve_document_trust(signed)
    assert capability is not None
    assert code_trust.execution_capability_allows(
        capability, (signed_entry, edited_entry)
    )

    for trust in (
        signed,
        external_document_trust(CodeTrustManifest("test", 1, ())),
        untrusted_document_trust(_manifest()),
    ):
        prospective, capability = code_trust.issue_local_edit_capability(
            trust,
            (unsigned_entry, edited_entry),
            edited_entries=(edited_entry,),
        )

        assert prospective == approve_document_trust(trust)
        assert capability is not None
        assert code_trust.execution_capability_allows(capability, (edited_entry,))
        assert not code_trust.execution_capability_allows(capability, (unsigned_entry,))

    for trust in (new_document_trust(), trusted_location_document_trust()):
        unchanged, capability = code_trust.issue_local_edit_capability(
            trust,
            (unsigned_entry, edited_entry),
            edited_entries=(edited_entry,),
        )

        assert unchanged == trust
        assert capability is not None


def test_local_edit_capability_does_not_promote_mixed_untrusted_content() -> None:
    external_entry = _entry(code="external()")
    edited_entry = _entry(code="local_edit()")
    untrusted = untrusted_document_trust(
        CodeTrustManifest("test", 1, (external_entry,))
    )

    prospective, capability = code_trust.issue_local_edit_capability(
        untrusted,
        (external_entry, edited_entry),
        edited_entries=(edited_entry,),
    )

    assert prospective == approve_document_trust(untrusted)
    assert capability is not None
    assert code_trust.execution_capability_allows(capability, (edited_entry,))
    assert not code_trust.execution_capability_allows(capability, (external_entry,))

    prospective, capability = code_trust.issue_local_edit_capability(
        untrusted,
        (edited_entry,),
        edited_entries=(edited_entry,),
    )

    assert prospective == approve_document_trust(untrusted)
    assert capability is not None
    assert code_trust.execution_capability_allows(capability, (edited_entry,))


def test_local_edit_capability_rejects_ambiguous_untrusted_content() -> None:
    edited_entry = _entry(code="same()", location="edited")
    external_entry = _entry(code="same()", location="external")
    untrusted = untrusted_document_trust(
        CodeTrustManifest("test", 1, (external_entry,))
    )

    prospective, capability = code_trust.issue_local_edit_capability(
        untrusted,
        (edited_entry, external_entry),
        edited_entries=(edited_entry,),
    )

    assert prospective == approve_document_trust(untrusted)
    assert not code_trust.execution_capability_allows(capability, (edited_entry,))
    assert not code_trust.execution_capability_allows(capability, (external_entry,))


def test_local_edit_capability_allows_single_untrusted_edit() -> None:
    edited_entry = _entry(code="same()", location="edited")
    external_entry = _entry(code="external()", location="external")
    untrusted = untrusted_document_trust(
        CodeTrustManifest("test", 1, (external_entry,))
    )

    prospective, capability = code_trust.issue_local_edit_capability(
        untrusted,
        (edited_entry,),
        edited_entries=(edited_entry,),
    )

    assert prospective == approve_document_trust(untrusted)
    assert code_trust.execution_capability_allows(capability, (edited_entry,))


def test_local_edit_commit_keeps_mixed_content_untrusted_but_runs_local_edit() -> None:
    external_entry = _entry(code="external()", location="external")
    edited_entry = _entry(code="local_edit()", location="edited")
    untrusted = untrusted_document_trust(
        CodeTrustManifest("test", 1, (external_entry,))
    )
    prospective, capability = code_trust.issue_local_edit_capability(
        untrusted,
        (edited_entry,),
        edited_entries=(edited_entry,),
    )

    assert prospective == approve_document_trust(untrusted)
    mixed = code_trust.commit_local_edit_trust(
        untrusted,
        capability,
        (external_entry,),
        (external_entry, edited_entry),
        edited_entries=(edited_entry,),
    )

    assert code_trust.document_trust_needs_review(mixed)
    assert code_trust.authorize_document_execution(mixed, (edited_entry,))[1]
    assert not code_trust.authorize_document_execution(mixed, (external_entry,))[1]
    assert not code_trust.authorize_document_execution(
        mixed, (external_entry, edited_entry)
    )[1]
    unchanged, mixed_capability = code_trust.issue_execution_capability(
        mixed, (external_entry, edited_entry)
    )
    assert unchanged == mixed
    assert mixed_capability is not None
    assert code_trust.execution_capability_allows(mixed_capability, (edited_entry,))
    assert not code_trust.execution_capability_allows(
        mixed_capability, (external_entry,)
    )
    unchanged, complete_capability = code_trust.issue_complete_execution_capability(
        mixed, (external_entry, edited_entry)
    )
    assert unchanged == mixed
    assert complete_capability is None
    _unchanged, local_capability = code_trust.issue_complete_execution_capability(
        mixed, (edited_entry,)
    )
    assert code_trust.execution_capability_allows(local_capability, (edited_entry,))
    assert (
        code_trust.commit_local_edit_trust(
            untrusted,
            capability,
            (external_entry,),
            (edited_entry,),
            edited_entries=(edited_entry,),
        )
        == prospective
    )


def test_cumulative_local_edits_promote_after_all_external_entries_are_replaced() -> (
    None
):
    first_external = _entry(code="first_external()", location="first")
    second_external = _entry(code="second_external()", location="second")
    first_local = _entry(code="first_local()", location="first")
    second_local = _entry(code="second_local()", location="second")
    untrusted = untrusted_document_trust(
        CodeTrustManifest("test", 1, (first_external, second_external))
    )

    _, first_capability = code_trust.issue_local_edit_capability(
        untrusted,
        (first_local,),
        edited_entries=(first_local,),
    )
    mixed = code_trust.commit_local_edit_trust(
        untrusted,
        first_capability,
        (first_external, second_external),
        (first_local, second_external),
        edited_entries=(first_local,),
    )

    assert code_trust.document_trust_needs_review(mixed)
    assert code_trust.authorize_document_execution(mixed, (first_local,))[1]
    _, second_capability = code_trust.issue_local_edit_capability(
        mixed,
        (first_local, second_local),
        edited_entries=(second_local,),
    )
    assert second_capability is not None
    promoted = code_trust.commit_local_edit_trust(
        mixed,
        second_capability,
        (first_local, second_external),
        (first_local, second_local),
        edited_entries=(second_local,),
    )

    assert document_trust_has_trusted_lineage(promoted)
    assert code_trust.authorize_document_execution(
        promoted, (first_local, second_local)
    )[1]


def test_equal_code_at_two_locations_requires_two_local_edits() -> None:
    external_a = _entry(code="same()", location="a")
    external_b = _entry(code="same()", location="b")
    untrusted = untrusted_document_trust(
        CodeTrustManifest("test", 1, (external_a, external_b))
    )

    _, capability_b = code_trust.issue_local_edit_capability(
        untrusted,
        (external_b,),
        edited_entries=(external_b,),
    )
    mixed = code_trust.commit_local_edit_trust(
        untrusted,
        capability_b,
        (external_a, external_b),
        (external_a, external_b),
        edited_entries=(external_b,),
    )

    assert code_trust.document_trust_needs_review(mixed)
    assert not code_trust.authorize_document_execution(mixed, (external_a,))[1]
    assert code_trust.authorize_document_execution(mixed, (external_b,))[1]
    _, combined_capability = code_trust.issue_execution_capability(
        mixed, (external_a, external_b)
    )
    assert combined_capability is None

    _, capability_a = code_trust.issue_local_edit_capability(
        mixed,
        (external_a,),
        edited_entries=(external_a,),
    )
    promoted = code_trust.commit_local_edit_trust(
        mixed,
        capability_a,
        (external_a, external_b),
        (external_a, external_b),
        edited_entries=(external_a,),
    )

    assert document_trust_has_trusted_lineage(promoted)


def test_relocation_is_not_adopted_as_a_local_edit() -> None:
    external = _entry(code="external()", location="a")
    relocated = _entry(code="external()", location="b")
    untrusted = untrusted_document_trust(CodeTrustManifest("test", 1, (external,)))
    _, capability = code_trust.issue_local_edit_capability(
        untrusted,
        (),
        edited_entries=(),
    )

    committed = code_trust.commit_local_edit_trust(
        untrusted,
        capability,
        (external,),
        (relocated,),
        edited_entries=(),
    )

    assert code_trust.document_trust_needs_review(committed)
    assert not code_trust.authorize_document_execution(committed, (relocated,))[1]


def test_relocating_changed_signed_source_manifest_fails_closed() -> None:
    signed_manifest = CodeTrustManifest(
        "test", 1, (_entry(code="signed()", location="feature"),)
    )
    changed_manifest = CodeTrustManifest(
        "test", 1, (_entry(code="changed()", location="feature"),)
    )

    relocated = code_trust.relocate_document_trust(
        _signed_trust(signed_manifest),
        changed_manifest,
        location_prefix="tools/0",
    )

    assert code_trust.document_trust_needs_review(relocated)
    assert not code_trust.authorize_document_execution(
        relocated, relocated.manifest.entries
    )[1]


def test_local_edit_commit_prunes_removed_local_identities() -> None:
    external_entry = _entry(code="external()", location="external")
    local_entry = _entry(code="local()", location="local")
    untrusted = untrusted_document_trust(
        CodeTrustManifest("test", 1, (external_entry,))
    )
    _, capability = code_trust.issue_local_edit_capability(
        untrusted,
        (local_entry,),
        edited_entries=(local_entry,),
    )
    mixed = code_trust.commit_local_edit_trust(
        untrusted,
        capability,
        (external_entry,),
        (external_entry, local_entry),
        edited_entries=(local_entry,),
    )

    _, delete_capability = code_trust.issue_local_edit_capability(
        mixed,
        (),
        edited_entries=(),
    )
    after_delete = code_trust.commit_local_edit_trust(
        mixed,
        delete_capability,
        (external_entry, local_entry),
        (external_entry,),
        edited_entries=(),
    )

    assert code_trust.document_trust_needs_review(after_delete)
    assert not code_trust.authorize_document_execution(after_delete, (local_entry,))[1]


def test_cancelled_local_edit_does_not_change_partial_trust() -> None:
    external_entry = _entry(code="external()", location="external")
    local_entry = _entry(code="local()", location="local")
    candidate = _entry(code="candidate()", location="candidate")
    untrusted = untrusted_document_trust(
        CodeTrustManifest("test", 1, (external_entry,))
    )
    _, local_capability = code_trust.issue_local_edit_capability(
        untrusted,
        (local_entry,),
        edited_entries=(local_entry,),
    )
    mixed = code_trust.commit_local_edit_trust(
        untrusted,
        local_capability,
        (external_entry,),
        (external_entry, local_entry),
        edited_entries=(local_entry,),
    )

    prospective, candidate_capability = code_trust.issue_local_edit_capability(
        mixed,
        (candidate,),
        edited_entries=(candidate,),
    )
    cancelled = code_trust.commit_local_edit_trust(
        mixed,
        candidate_capability,
        (external_entry, local_entry),
        (external_entry, local_entry),
        edited_entries=(candidate,),
    )

    assert prospective != mixed
    assert cancelled == mixed
    assert code_trust.authorize_document_execution(cancelled, (local_entry,))[1]
    assert not code_trust.authorize_document_execution(cancelled, (candidate,))[1]


def test_local_edit_commit_adopts_transaction_derived_identities() -> None:
    external_entry = _entry(code="external()", location="external")
    edited_entry = _entry(code="edited()", location="edited")
    derived_entry = _entry(code="derived()", location="derived")
    untrusted = untrusted_document_trust(
        CodeTrustManifest("test", 1, (external_entry,))
    )
    _, capability = code_trust.issue_local_edit_capability(
        untrusted,
        (edited_entry,),
        edited_entries=(edited_entry,),
    )

    mixed = code_trust.commit_local_edit_trust(
        untrusted,
        capability,
        (external_entry,),
        (external_entry, edited_entry, derived_entry),
        edited_entries=(edited_entry,),
    )

    assert code_trust.document_trust_needs_review(mixed)
    assert code_trust.authorize_document_execution(
        mixed, (edited_entry, derived_entry)
    )[1]
    assert not code_trust.authorize_document_execution(mixed, (external_entry,))[1]


def test_local_edit_commit_stores_the_post_commit_manifest() -> None:
    external_entry = _entry(code="external()", location="external")
    local_entry = _entry(code="local()", location="local")
    original_manifest = CodeTrustManifest("test", 1, (external_entry,))
    current_manifest = CodeTrustManifest("test", 1, (external_entry, local_entry))
    untrusted = untrusted_document_trust(original_manifest)
    _, capability = code_trust.issue_local_edit_capability(
        untrusted,
        (local_entry,),
        edited_entries=(local_entry,),
    )

    committed = code_trust.commit_local_edit_trust(
        untrusted,
        capability,
        original_manifest.entries,
        current_manifest.entries,
        edited_entries=(local_entry,),
        document_manifest=current_manifest,
    )

    assert committed.manifest == current_manifest
    assert code_trust.document_trust_needs_review(committed)
    assert code_trust.authorize_document_execution(committed, (local_entry,))[1]
    assert not code_trust.authorize_document_execution(committed, (external_entry,))[1]


def test_non_replacing_import_preserves_only_exact_known_identities() -> None:
    signed_entry = _entry(code="signed()", location="signed")
    external_entry = _entry(code="external()", location="external")
    signed = _signed_trust(CodeTrustManifest("test", 1, (signed_entry,)))
    external = untrusted_document_trust(CodeTrustManifest("test", 1, (external_entry,)))

    combined = merge_document_trust(signed, external, replace=False)

    assert code_trust.document_trust_needs_review(combined)
    assert code_trust.authorize_document_execution(combined, (signed_entry,))[1]
    assert not code_trust.authorize_document_execution(combined, (external_entry,))[1]
    replaced = merge_document_trust(combined, external, replace=True)
    assert not code_trust.authorize_document_execution(replaced, (signed_entry,))[1]


def test_local_edit_commit_rejects_an_edit_outside_the_validation_capability() -> None:
    candidate = _entry(code="candidate()")
    omitted_edit = _entry(code="omitted()")
    external = external_document_trust(CodeTrustManifest("test", 1, ()))
    prospective, capability = code_trust.issue_local_edit_capability(
        external,
        (candidate,),
        edited_entries=(candidate, omitted_edit),
    )

    assert prospective == approve_document_trust(external)
    assert (
        code_trust.commit_local_edit_trust(
            external,
            capability,
            (),
            (candidate,),
            edited_entries=(candidate, omitted_edit),
        )
        == external
    )


def test_local_edit_commit_accepts_entries_derived_during_validation() -> None:
    saved_entry = _entry(code="saved_model()", location="model")
    edited_entry = _entry(code="edited_model()", location="model")
    derived_entry = _entry(code="p1_center - p0_center", location="parameters/p1")
    signed = _signed_trust(CodeTrustManifest("test", 1, (saved_entry,)))
    _, capability = code_trust.issue_local_edit_capability(
        signed,
        (edited_entry,),
        edited_entries=(edited_entry,),
    )

    committed = code_trust.commit_local_edit_trust(
        signed,
        capability,
        (saved_entry,),
        (edited_entry, derived_entry),
        edited_entries=(edited_entry,),
    )

    assert committed == approve_document_trust(signed)


def test_local_edit_capability_is_bound_to_the_candidate_inventory() -> None:
    candidate = _entry(code="local_edit()")
    omitted_edit = _entry(code="not_in_candidate()")

    prospective, capability = code_trust.issue_local_edit_capability(
        new_document_trust(),
        (candidate,),
        edited_entries=(candidate, omitted_edit),
    )

    assert prospective == new_document_trust()
    assert capability is not None
    assert code_trust.execution_capability_allows(capability, (candidate,))
    assert not code_trust.execution_capability_allows(capability, (omitted_edit,))
    assert not code_trust.execution_capability_allows(
        capability, (_entry(code="changed_after_authorization()"),)
    )


def test_local_edit_capability_handles_empty_and_deletion_edits() -> None:
    retained = _entry(code="retained()")
    removed = _entry(code="removed()")
    signed = _signed_trust(CodeTrustManifest("test", 1, (retained, removed)))

    prospective, capability = code_trust.issue_local_edit_capability(
        signed,
        (retained,),
        edited_entries=(),
    )

    assert prospective == approve_document_trust(signed)
    assert capability is not None
    assert code_trust.execution_capability_allows(capability, (retained,))
    assert not code_trust.execution_capability_allows(capability, (removed,))

    safe_external = external_document_trust(CodeTrustManifest("test", 1, ()))
    untrusted = untrusted_document_trust(_manifest())
    for trust, expected in (
        (signed, approve_document_trust(signed)),
        (safe_external, approve_document_trust(safe_external)),
        (untrusted, approve_document_trust(untrusted)),
    ):
        prospective, capability = code_trust.issue_local_edit_capability(
            trust,
            (),
            edited_entries=(),
        )

        assert prospective == expected
        assert capability is not None
        assert code_trust.execution_capability_allows(capability, ())


def test_signed_execution_identity_is_immutable_after_verification() -> None:
    manifest = _manifest()
    signed = _signed_trust(manifest)
    original = _entry(location="runtime/window/0")
    unchanged, capability = code_trust.issue_execution_capability(signed, (original,))

    manifest.entries[0].context["enabled"] = False
    original.context["enabled"] = False

    assert unchanged == signed
    assert capability is not None
    assert not code_trust.authorize_document_execution(signed, (manifest.entries[0],))[
        1
    ]
    assert not code_trust.execution_capability_allows(capability, (original,))
    recreated = _entry(location="another/window/0")
    assert code_trust.authorize_document_execution(signed, (recreated,))[1]
    assert code_trust.execution_capability_allows(capability, (recreated,))


def test_local_document_policy_does_not_build_entries_on_the_fast_path() -> None:
    calls: list[None] = []

    def entries():
        calls.append(None)
        return _manifest().entries

    for trust in (
        new_document_trust(),
        approve_document_trust(untrusted_document_trust(_manifest())),
        trusted_location_document_trust(),
    ):
        unchanged, allowed = code_trust.authorize_document_execution(trust, entries)
        assert allowed
        assert unchanged == trust
    assert calls == []


def test_opaque_code_payload_metadata_binds_exact_bytes() -> None:
    entry = code_trust.create_payload_entry(
        "test.serialized-callable",
        "payload/result",
        "Serialized callable",
        b"original payload",
    )
    attrs: dict[str, object] = {}
    store_code_payload_entries(attrs, (entry,))

    assert code_payload_entries_from_metadata(attrs) == (entry,)

    manifest = CodeTrustManifest("test.payload", 1, (entry,))
    signed = _signed_trust(manifest)
    assert (
        code_trust.verify_document_payload_entries(signed, (entry,), (entry,)) == signed
    )
    changed_entry = code_trust.create_payload_entry(
        "test.serialized-callable",
        "payload/result",
        "Serialized callable",
        b"changed payload",
    )
    assert not document_trust_is_trusted(
        code_trust.verify_document_payload_entries(
            signed,
            (entry,),
            (changed_entry,),
        )
    )
    assert not document_trust_is_trusted(
        code_trust.verify_document_payload_entries(
            new_document_trust(), (), (changed_entry,)
        )
    )
    assert document_trust_is_trusted(
        code_trust.verify_document_payload_entries(
            trusted_location_document_trust(), (), (changed_entry,)
        )
    )


def test_code_trust_notary_sign_check_remove_and_domain_separation(tmp_path) -> None:
    notary = CodeTrustNotary(tmp_path)
    manifest = _manifest()

    assert not notary.check(manifest)
    notary.sign(manifest)
    assert notary.check(manifest)
    assert not notary.check(_manifest(code="value = 2"))
    assert not notary.check(_manifest(domain="erlab.figure-composer-file"))
    assert not notary.check(_manifest(version=2))
    notary.remove(manifest)
    assert not notary.check(manifest)


def test_code_trust_notary_reset_and_culling(tmp_path) -> None:
    notary = CodeTrustNotary(tmp_path, cache_size=4)
    manifests = tuple(_manifest(code=f"value = {index}") for index in range(5))
    for manifest in manifests:
        notary.sign(manifest)

    assert not notary.check(manifests[0])
    assert not notary.check(manifests[1])
    assert all(notary.check(manifest) for manifest in manifests[2:])

    notary.reset(domain="erlab.workspace")
    assert not notary.check(manifests[2])


def test_code_trust_notary_secret_and_database_tampering_fail_closed(tmp_path) -> None:
    notary = CodeTrustNotary(tmp_path)
    manifest = _manifest()
    notary.sign(manifest)
    secret_path = tmp_path / "code_trust_secret"
    database_path = tmp_path / "code_signatures.db"

    secret_path.write_bytes(os.urandom(1024))
    assert not notary.check(manifest)

    with closing(sqlite3.connect(database_path)) as connection:
        connection.execute(
            """
            INSERT INTO signatures (domain, algorithm, signature, last_seen)
            VALUES (?, ?, ?, ?)
            """,
            (manifest.domain, "sha256", "0" * 64, 0.0),
        )
        connection.commit()
    assert not notary.check(manifest)


def test_application_code_trust_storage_failure_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest()

    def unavailable_notary() -> typing.NoReturn:
        raise RuntimeError("storage unavailable")

    monkeypatch.setattr(
        _application,
        "_application_code_trust_notary",
        unavailable_notary,
    )

    loaded = _application.load_document_trust(manifest)
    assert not document_trust_is_trusted(loaded)

    saved, signature_stored = _application.save_document_trust(
        new_document_trust(),
        manifest,
    )
    assert not signature_stored
    assert document_trust_has_trusted_lineage(saved)
    with pytest.raises(RuntimeError, match="storage unavailable"):
        _application.reset_saved_code_trust()


def test_application_does_not_sign_unregistered_manifest_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _entry(code="first()")
    second = _entry(code="second()")
    original = CodeTrustManifest("test", 1, (first, second))
    changed_manifests = (
        CodeTrustManifest("other", 1, (first, second)),
        CodeTrustManifest("test", 2, (first, second)),
        CodeTrustManifest(
            "test", 1, (_entry(code="first()", location="other"), second)
        ),
        CodeTrustManifest("test", 1, (second, first)),
        CodeTrustManifest("test", 1, (_entry(code="changed()"), second)),
    )
    signed_manifests: list[CodeTrustManifest] = []

    class Notary:
        def sign(self, manifest: CodeTrustManifest) -> None:
            signed_manifests.append(manifest)

    monkeypatch.setattr(_application, "_application_code_trust_notary", Notary)

    for changed_manifest in changed_manifests:
        saved, signature_stored = _application.save_document_trust(
            _signed_trust(original),
            changed_manifest,
            saved_trusted_lineage=True,
        )

        assert signature_stored
        assert not document_trust_is_trusted(saved)
    assert signed_manifests == []

    saved, signature_stored = _application.save_document_trust(
        _signed_trust(original),
        original,
        saved_trusted_lineage=True,
    )

    assert signature_stored
    assert document_trust_is_trusted(saved)
    assert signed_manifests == [original]

    mutable_manifest = _manifest()
    mutable_trust = _signed_trust(mutable_manifest)
    mutable_manifest.entries[0].context["enabled"] = False
    saved, signature_stored = _application.save_document_trust(
        mutable_trust,
        mutable_manifest,
        saved_trusted_lineage=True,
    )

    assert signature_stored
    assert not document_trust_is_trusted(saved)
    assert signed_manifests == [original]


def test_untrusted_save_preserves_local_identities_only_in_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    external_entry = _entry(code="external()", location="external")
    local_entry = _entry(code="local()", location="local")
    original_manifest = CodeTrustManifest("test", 1, (external_entry,))
    saved_manifest = CodeTrustManifest("test", 1, (external_entry, local_entry))
    untrusted = untrusted_document_trust(original_manifest)
    _, capability = code_trust.issue_local_edit_capability(
        untrusted,
        (local_entry,),
        edited_entries=(local_entry,),
    )
    mixed = code_trust.commit_local_edit_trust(
        untrusted,
        capability,
        original_manifest.entries,
        saved_manifest.entries,
        edited_entries=(local_entry,),
    )
    signed_manifests: list[CodeTrustManifest] = []

    class Notary:
        def sign(self, manifest: CodeTrustManifest) -> None:
            signed_manifests.append(manifest)

        def check(self, manifest: CodeTrustManifest) -> bool:
            return False

    monkeypatch.setattr(_application, "_application_code_trust_notary", Notary)

    saved, _signature_stored = _application.save_document_trust(
        mixed,
        saved_manifest,
        saved_trusted_lineage=False,
    )

    assert signed_manifests == []
    assert code_trust.document_trust_needs_review(saved)
    assert code_trust.authorize_document_execution(saved, (local_entry,))[1]
    assert not code_trust.authorize_document_execution(saved, (external_entry,))[1]
    reopened = _application.load_document_trust(saved_manifest)
    assert code_trust.document_trust_needs_review(reopened)
    assert not code_trust.authorize_document_execution(reopened, (local_entry,))[1]

    pruned, _signature_stored = _application.save_document_trust(
        saved,
        original_manifest,
        saved_trusted_lineage=False,
    )
    assert not code_trust.authorize_document_execution(pruned, (local_entry,))[1]


def test_application_code_trust_directory_is_stable_across_qt_hosts(
    monkeypatch: pytest.MonkeyPatch,
    qapp,
) -> None:
    monkeypatch.delenv("ERLAB_CODE_TRUST_DIRECTORY")
    previous_application = qapp.applicationName()
    previous_organization = qapp.organizationName()
    previous_organization_domain = qapp.organizationDomain()
    try:
        qapp.setOrganizationName("")
        qapp.setOrganizationDomain("")
        qapp.setApplicationName("ImageTool Manager")
        legacy_manager_directory = (
            pathlib.Path(
                QtCore.QStandardPaths.writableLocation(
                    QtCore.QStandardPaths.StandardLocation.AppDataLocation
                )
            )
            / "code-trust"
        )
        manager_directory = _application.application_code_trust_directory()

        host_directories = []
        for organization, organization_domain, application in (
            ("First Host Organization", "first.example", "First Host"),
            ("Second Host Organization", "second.example", "Second Host"),
        ):
            qapp.setOrganizationName(organization)
            qapp.setOrganizationDomain(organization_domain)
            qapp.setApplicationName(application)
            host_directories.append(_application.application_code_trust_directory())
    finally:
        qapp.setOrganizationName(previous_organization)
        qapp.setOrganizationDomain(previous_organization_domain)
        qapp.setApplicationName(previous_application)

    assert manager_directory == legacy_manager_directory
    assert host_directories == [manager_directory, manager_directory]


def test_application_code_trust_directory_honors_environment_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    configured = tmp_path / "custom-trust"
    monkeypatch.setenv("ERLAB_CODE_TRUST_DIRECTORY", str(configured))

    assert _application.application_code_trust_directory() == configured


def test_code_trust_notary_missing_secret_invalidates_saved_signatures(
    tmp_path,
) -> None:
    notary = CodeTrustNotary(tmp_path)
    manifest = _manifest()
    notary.sign(manifest)
    secret_path = tmp_path / "code_trust_secret"

    secret_path.unlink()

    assert not notary.check(manifest)
    assert len(secret_path.read_bytes()) == 1024


def test_code_trust_notary_failed_secret_write_can_recover(
    tmp_path, monkeypatch
) -> None:
    notary = CodeTrustNotary(tmp_path)
    manifest = _manifest()

    def fail_fsync(_descriptor: int) -> None:
        raise OSError("write failed")

    with monkeypatch.context() as patch:
        patch.setattr(os, "fsync", fail_fsync)
        with pytest.raises(CodeTrustError, match="Could not create"):
            notary.sign(manifest)

    assert not (tmp_path / "code_trust_secret").exists()
    assert not tuple(tmp_path.glob(".code_trust_secret-*"))

    notary.sign(manifest)

    assert notary.check(manifest)


def test_code_trust_notary_replaces_corrupt_database_fail_closed(tmp_path) -> None:
    notary = CodeTrustNotary(tmp_path)
    manifest = _manifest()
    notary.sign(manifest)
    database_path = tmp_path / "code_signatures.db"
    database_path.write_bytes(b"not a sqlite database")

    assert not notary.check(manifest)
    assert (
        tmp_path / "code_signatures.db.bak"
    ).read_bytes() == b"not a sqlite database"


def test_code_trust_notary_does_not_replace_locked_database(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    notary = CodeTrustNotary(tmp_path)
    manifest = _manifest()
    notary.sign(manifest)
    database_path = tmp_path / "code_signatures.db"
    original_database = database_path.read_bytes()

    def raise_locked(_connection: sqlite3.Connection) -> None:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(notary, "_initialize_database", raise_locked)

    with pytest.raises(CodeTrustError, match="Could not access"):
        notary._connect()
    assert database_path.read_bytes() == original_database
    assert not (tmp_path / "code_signatures.db.bak").exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits only")
def test_code_trust_notary_uses_private_permissions(tmp_path) -> None:
    trust_directory = tmp_path / "trust"
    notary = CodeTrustNotary(trust_directory)
    notary.sign(_manifest())

    assert (trust_directory / "code_trust_secret").stat().st_mode & 0o777 == 0o600
    assert (trust_directory / "code_signatures.db").stat().st_mode & 0o777 == 0o600
    assert trust_directory.stat().st_mode & 0o777 == 0o700


def test_trusted_location_policy_resolves_descendants_and_symlinks(tmp_path) -> None:
    trusted = tmp_path / "research"
    outside = tmp_path / "outside"
    trusted.mkdir()
    outside.mkdir()
    workspace = trusted / "nested" / "figure.itws"
    workspace.parent.mkdir()
    workspace.touch()
    outside_workspace = outside / "outside.itws"
    outside_workspace.touch()
    (trusted / "escape").symlink_to(outside, target_is_directory=True)

    locations = (("erlab.workspace", trusted),)

    assert document_path_is_trusted("erlab.workspace", workspace, locations)
    assert not document_path_is_trusted("other", workspace, locations)
    assert not document_path_is_trusted(
        "erlab.workspace",
        trusted / "escape/outside.itws",
        locations,
    )


def test_validate_trusted_location_rejects_filesystem_root(tmp_path) -> None:
    folder = tmp_path / "research"
    folder.mkdir()

    assert validate_trusted_location(folder) == folder.resolve()
    with pytest.raises(ValueError, match="filesystem root"):
        validate_trusted_location(folder.anchor)
