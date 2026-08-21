"""Authorize saved documents that contain user-provided Python.

This private package owns the security policy. Feature code describes executable
content with :func:`create_entry`. A document host builds one ordered manifest and owns
one opaque trust value. The feature must not inspect trust reasons, signatures, or
lineage.

An ``.itws`` workspace has four execution boundaries:

1. Provenance passes a recorded script to ``exec``.
2. Figure Composer passes custom Python to ``exec`` or ``eval``.
3. A fitting tool or provenance replay decodes an lmfit model or result that contains
   custom Python.
4. lmfit evaluates a saved parameter expression.

Guard the call that executes or decodes this content. File paths, loaders, tool names,
numeric arrays, previews, and declarative provenance are not trust boundaries. Safe
work must continue when executable content is paused.

Each entry contains a stable feature name, a document location, exact code, and the
JSON-compatible context that changes execution. Include such values as enabled state,
mode, inputs, and order. Exclude labels and unrelated interface state. A saved manifest
includes disabled code because a user can enable it later. Opaque lmfit entries also
contain a SHA-256 digest of their executable payload fragments.

A saved signature covers the complete canonical manifest. The domain separates file
formats. The policy version invalidates signatures after execution semantics change.
Document identifiers, paths, and serialized trust flags are never proof of trust.
The manifest bound to a trust value is only the in-memory baseline for the next
transaction. It is not trust proof. Binding changed content to signature trust revokes
the signature instead of transferring it.

To add an executable feature:

* Keep a small entry producer beside the feature. Do not scan object graphs or use a
  contributor registry.
* Include all stored content that can reach an execution boundary.
* Build the same entry immediately before execution.
* Ask the document host for authorization. Normal execution requests receive a
  capability only when every requested entry is allowed. A graph executor can use
  :func:`issue_execution_capability` directly and check each entry at its boundary.
  This low-level capability can allow only a subset in a mixed document.
  :func:`execute_with_capability` keeps the final check and the guarded call together.
* Stop only the unauthorized execution.

An explicit user edit uses :func:`issue_local_edit_capability`. The feature supplies
the candidate execution inventory and the exact entries that the user added or
changed. The capability allows those entries and any entries already allowed by the
stored-content policy. It remains an allow-list, so other external entries in the same
graph stay paused. The returned trust state is prospective. Apply it only after
candidate validation and the edit commit both succeed. A cancelled or failed edit must
leave the document trust unchanged. Deletion is also an explicit edit, even when the
edited-entry inventory is empty. After commit, the host compares the complete document
manifest before and after the transaction. The framework adds explicit and derived
current document identities to an ephemeral local allow-list and removes identities
that are no longer present. A document identity includes the exact entry location.
Execution capabilities omit the location so they remain valid after a runtime path
translation. The document remains untrusted while any current entry is outside the
prior policy and this allow-list. During validation, the host can expose the capability,
but it must not expose the prospective local lineage. After commit, pass the current
manifest to :func:`commit_local_edit_trust` or bind it with
:func:`bind_document_trust_manifest` for use as the next transaction baseline.

An explicitly selected external executable file is a separate case. Review an
isolated manifest for the candidate, issue a capability for only its exact entries,
and decode the exact reviewed bytes. Commit the resulting document state as a local
edit only after decoding and validation succeed.

For example::

    def code_trust_entries(state):
        return (
            create_entry(
                "erlab.example.expression",
                "operations/0",
                state.expression,
                {"enabled": state.enabled, "mode": state.mode},
            ),
        )

New local documents have trusted lineage. External documents do not. Explicit approval
grants local lineage. A successful save signs only a committed manifest with trusted
lineage. A mixed untrusted save keeps its local allow-list only in memory. Reopening it
requires review again. Imports combine lineage, and untrusted executable content makes
the result untrusted. A non-replacing import can retain exact known local identities.
A replacement discards identities from the replaced document. Trusted folders are an
explicit application policy that bypasses signature checks for matching resolved
paths.

A standalone ImageTool is not a document-trust host. It does not run executable stored
provenance. ImageTool Manager injects the stored-code authorizer when it owns the
provenance document.

Keep future integration at these entry, authorization, and document-lifecycle seams.
Do not add feature lifecycle state to this package. This is not a public API.
"""

from erlab.interactive._code_trust._api import (
    approve_document_trust,
    authorize_document_execution,
    bind_document_trust_manifest,
    commit_local_edit_trust,
    create_entry,
    create_manifest,
    document_trust_description,
    document_trust_has_trusted_lineage,
    document_trust_is_trusted,
    document_trust_needs_review,
    execute_with_capability,
    execution_capability_allows,
    external_document_trust,
    issue_complete_execution_capability,
    issue_execution_capability,
    issue_local_edit_capability,
    manifest_has_code,
    manifest_review_text,
    merge_document_trust,
    new_document_trust,
    relocate_document_trust,
    relocate_manifest_entries,
    trusted_location_document_trust,
    untrusted_document_trust,
    verify_document_payload_entries,
)
from erlab.interactive._code_trust._application import reset_saved_code_trust
from erlab.interactive._code_trust._locations import (
    document_path_is_trusted,
    validate_trusted_location,
)
from erlab.interactive._code_trust._payloads import create_payload_entry

__all__ = [
    "approve_document_trust",
    "authorize_document_execution",
    "bind_document_trust_manifest",
    "commit_local_edit_trust",
    "create_entry",
    "create_manifest",
    "create_payload_entry",
    "document_path_is_trusted",
    "document_trust_description",
    "document_trust_has_trusted_lineage",
    "document_trust_is_trusted",
    "document_trust_needs_review",
    "execute_with_capability",
    "execution_capability_allows",
    "external_document_trust",
    "issue_complete_execution_capability",
    "issue_execution_capability",
    "issue_local_edit_capability",
    "manifest_has_code",
    "manifest_review_text",
    "merge_document_trust",
    "new_document_trust",
    "relocate_document_trust",
    "relocate_manifest_entries",
    "reset_saved_code_trust",
    "trusted_location_document_trust",
    "untrusted_document_trust",
    "validate_trusted_location",
    "verify_document_payload_entries",
]
