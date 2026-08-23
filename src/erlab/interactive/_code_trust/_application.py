"""Application storage adapter for the general code-trust notary."""

from __future__ import annotations

import functools
import logging
import os
import pathlib
import typing

from qtpy import QtCore

from erlab.interactive._code_trust._api import (
    _document_trust,
    _document_trust_after_save,
    _DocumentTrust,
    document_trust_has_trusted_lineage,
    manifest_has_code,
)
from erlab.interactive._code_trust._core import CodeTrustReason
from erlab.interactive._code_trust._notary import CodeTrustError, CodeTrustNotary

if typing.TYPE_CHECKING:
    from erlab.interactive._code_trust._core import CodeTrustManifest

_CODE_TRUST_DIRECTORY_ENV_VAR = "ERLAB_CODE_TRUST_DIRECTORY"
_APPLICATION_DATA_DIRECTORY_NAME = "ImageTool Manager"
logger = logging.getLogger(__name__)


def application_code_trust_directory() -> pathlib.Path:
    """Return the stable private directory used for code-trust state.

    ``GenericDataLocation/ImageTool Manager`` is independent of the Qt host's
    application and organization names. It is also the path that
    ``AppDataLocation`` returns for ImageTool Manager when the organization name is
    empty. Existing approvals from the standalone manager therefore stay available.
    """
    if configured := os.environ.get(_CODE_TRUST_DIRECTORY_ENV_VAR):
        return pathlib.Path(configured)
    location = QtCore.QStandardPaths.writableLocation(
        QtCore.QStandardPaths.StandardLocation.GenericDataLocation
    )
    if not location:
        raise RuntimeError("Could not determine the code trust storage directory")
    return pathlib.Path(location) / _APPLICATION_DATA_DIRECTORY_NAME / "code-trust"


@functools.cache
def _application_code_trust_notary() -> CodeTrustNotary:
    """Return the process-wide code-trust notary."""
    return CodeTrustNotary(application_code_trust_directory())


def reset_saved_code_trust(*, domain: str | None = None) -> None:
    """Remove saved executable-content signatures."""
    try:
        _application_code_trust_notary().reset(domain=domain)
    except (CodeTrustError, RuntimeError) as exc:
        raise RuntimeError(str(exc)) from exc


def _code_trust_reason(manifest: CodeTrustManifest) -> CodeTrustReason:
    """Return a fail-closed reason when application storage is unavailable."""
    if not manifest_has_code(manifest):
        return CodeTrustReason.NO_EXECUTABLE_CODE
    try:
        signed = _application_code_trust_notary().check(manifest)
    except (CodeTrustError, RuntimeError):
        logger.warning("Could not read durable code trust", exc_info=True)
        signed = False
    return CodeTrustReason.SIGNATURE if signed else CodeTrustReason.UNTRUSTED


def load_document_trust(manifest: CodeTrustManifest) -> _DocumentTrust:
    """Load complete trust state for one application document manifest."""
    return _document_trust(_code_trust_reason(manifest), manifest)


def load_imported_document_trust(
    document_manifest: CodeTrustManifest,
    imported_manifest: CodeTrustManifest,
) -> _DocumentTrust:
    """Load trust for selected content from a complete saved document.

    A saved signature covers the complete document manifest. Selected imports retain
    that authorization. An unsigned source can still use an approval for the exact
    selected manifest when one exists.
    """
    reason = _code_trust_reason(document_manifest)
    remaining_entries = iter(document_manifest.entries)
    is_ordered_subset = (
        imported_manifest.domain == document_manifest.domain
        and imported_manifest.policy_version == document_manifest.policy_version
        and all(
            any(candidate == entry for candidate in remaining_entries)
            for entry in imported_manifest.entries
        )
    )
    if reason == CodeTrustReason.SIGNATURE and is_ordered_subset:
        return _document_trust(reason, imported_manifest)
    return load_document_trust(imported_manifest)


def save_document_trust(
    trust: _DocumentTrust,
    manifest: CodeTrustManifest,
    *,
    saved_trusted_lineage: bool | None = None,
) -> tuple[_DocumentTrust, bool]:
    """Store durable trust and return the state for committed content."""
    if saved_trusted_lineage is None:
        saved_trusted_lineage = document_trust_has_trusted_lineage(trust)
    if (
        trust.reason == CodeTrustReason.SIGNATURE
        and trust.manifest_identity != manifest.canonical_bytes()
    ):
        saved_trusted_lineage = False
        trust = _document_trust(CodeTrustReason.UNTRUSTED, manifest)
    signature_stored = True
    if manifest_has_code(manifest) and saved_trusted_lineage:
        try:
            _application_code_trust_notary().sign(manifest)
        except (CodeTrustError, RuntimeError):
            logger.warning("Could not store durable code trust", exc_info=True)
            signature_stored = False
    return (
        _document_trust_after_save(
            trust,
            manifest,
            saved_trusted_lineage=saved_trusted_lineage,
            signature_stored=signature_stored,
        ),
        signature_stored,
    )
