"""Revision identity for standard Python extension entry points."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import typing
import urllib.parse

if typing.TYPE_CHECKING:
    import importlib.metadata
    from collections.abc import Mapping


class _EntryPointRevisionError(ValueError):
    """An installed entry point does not have usable revision metadata."""


def _editable_source_fingerprint(direct_url: Mapping[str, typing.Any]) -> str:
    """Hash Python sources in an editable distribution project."""
    url = direct_url.get("url")
    if not isinstance(url, str):
        raise _EntryPointRevisionError("Editable package URL is unavailable")
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("", "file"):
        raise _EntryPointRevisionError("Editable package URL is not a local path")
    root = pathlib.Path(urllib.parse.unquote(parsed.path))
    if not root.is_dir():
        raise _EntryPointRevisionError(
            f"Editable package directory is unavailable: {root}"
        )
    candidates: list[pathlib.Path] = []
    ignored_directories = {
        ".git",
        ".hg",
        ".mypy_cache",
        ".nox",
        ".pytest_cache",
        ".ruff_cache",
        ".svn",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "venv",
    }

    def walk_error(error: OSError) -> typing.Never:
        raise _EntryPointRevisionError(
            f"Could not inspect editable package directory {root}: {error}"
        ) from error

    for directory, directory_names, file_names in os.walk(root, onerror=walk_error):
        directory_names[:] = sorted(
            name for name in directory_names if name not in ignored_directories
        )
        for file_name in sorted(file_names):
            path = pathlib.Path(directory, file_name)
            if path.suffix in {".py", ".pyi"} or path.name == "pyproject.toml":
                candidates.append(path)
    if not candidates:
        raise _EntryPointRevisionError(
            f"Editable package contains no fingerprintable source: {root}"
        )
    digest = hashlib.sha256()
    for path in candidates:
        try:
            source = path.read_bytes()
        except OSError as error:
            raise _EntryPointRevisionError(
                f"Could not fingerprint editable package source {path}: {error}"
            ) from error
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(len(source).to_bytes(8, "big"))
        digest.update(source)
    return digest.hexdigest()


def _entry_point_revision_payload(
    entry_point: importlib.metadata.EntryPoint,
) -> tuple[str, str, str, bool]:
    """Return stable distribution metadata used to identify one revision."""
    distribution = entry_point.dist
    distribution_name = (
        entry_point.name
        if distribution is None
        else str(distribution.metadata.get("Name", entry_point.name))
    )
    distribution_version = "" if distribution is None else distribution.version
    editable = False
    editable_fingerprint: str | None = None
    if distribution is not None:
        direct_url = distribution.read_text("direct_url.json")
        if direct_url:
            try:
                parsed = json.loads(direct_url)
            except json.JSONDecodeError as error:
                raise _EntryPointRevisionError(
                    "Editable package metadata is not valid JSON"
                ) from error
            if not isinstance(parsed, dict):
                raise _EntryPointRevisionError(
                    "Editable package metadata must be a JSON object"
                )
            directory_info = parsed.get("dir_info", {})
            if not isinstance(directory_info, dict):
                raise _EntryPointRevisionError(
                    "Editable package directory metadata must be a JSON object"
                )
            editable = bool(directory_info.get("editable", False))
            if editable:
                editable_fingerprint = _editable_source_fingerprint(parsed)
    payload = json.dumps(
        {
            "group": entry_point.group,
            "name": entry_point.name,
            "value": entry_point.value,
            "distribution": distribution_name,
            "version": distribution_version,
            "editable": editable,
            "editable_source": editable_fingerprint,
        },
        sort_keys=True,
    )
    return distribution_name, distribution_version, payload, editable


def _entry_point_revision(entry_point: importlib.metadata.EntryPoint) -> str:
    """Return the exact revision hash for one entry point without importing it."""
    _name, _version, payload, _editable = _entry_point_revision_payload(entry_point)
    return hashlib.sha256(payload.encode()).hexdigest()
