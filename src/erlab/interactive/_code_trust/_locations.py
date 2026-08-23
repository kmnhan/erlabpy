"""Feature-neutral trusted-location matching for executable documents.

Folder trust is an explicit user decision. It uses resolved path containment and does
not try to classify filesystems as local or remote. A filesystem label cannot reliably
identify who can change a folder.
"""

from __future__ import annotations

import pathlib
import typing

if typing.TYPE_CHECKING:
    import os
    from collections.abc import Iterable


def _normalized_path(path: str | os.PathLike[str], *, strict: bool) -> pathlib.Path:
    return pathlib.Path(path).expanduser().resolve(strict=strict)


def validate_trusted_location(path: str | os.PathLike[str]) -> pathlib.Path:
    """Resolve and validate a folder before it enters trusted settings."""
    resolved = _normalized_path(path, strict=True)
    if not resolved.is_dir():
        raise ValueError("Trusted location must be a folder")
    root = pathlib.Path(resolved.anchor).resolve(strict=False)
    if resolved == root:
        raise ValueError("A filesystem root cannot be trusted")
    return resolved


def document_path_is_trusted(
    domain: str,
    document_path: str | os.PathLike[str],
    locations: Iterable[tuple[str, str | os.PathLike[str]]],
) -> bool:
    """Return whether a document is below a trusted folder for its domain."""
    try:
        document = _normalized_path(document_path, strict=True)
    except OSError:
        return False
    for location_domain, location_path in locations:
        if location_domain != domain:
            continue
        try:
            folder = validate_trusted_location(location_path)
        except (OSError, ValueError):
            continue
        if not document.is_relative_to(folder):
            continue
        return True
    return False
