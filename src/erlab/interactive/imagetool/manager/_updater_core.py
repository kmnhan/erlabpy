from __future__ import annotations

import dataclasses
import hashlib
import os
import pathlib
import platform
import re
import sys

import requests

import erlab
from erlab.interactive.imagetool.manager import _get_updater_settings

OWNER = "kmnhan"
REPO = "erlabpy"

# Regex patterns to select the right asset from a Release based on OS/arch.
ASSET_PATTERNS: dict[str, re.Pattern] = {
    "windows-amd64": re.compile(r"[-_ ]windows\.zip$", re.IGNORECASE),
    "macos-arm": re.compile(r".*[-_ ]mac(?:os)?[-_ ]arm\.zip$", re.IGNORECASE),
    "macos-intel": re.compile(r".*[-_ ]mac(?:os)?[-_ ]intel\.zip$", re.IGNORECASE),
}


@dataclasses.dataclass
class ReleaseAsset:
    name: str
    size: int
    download_url: str
    digest: str


@dataclasses.dataclass
class ReleaseInfo:
    tag: str
    body: str
    asset: ReleaseAsset


def runtime_platform() -> str:
    osname = platform.system()  # "Windows", "Darwin", "Linux"
    machine = platform.machine() or platform.processor()
    # Normalize a few common aliases
    m = machine.lower()
    if m in {"amd64", "x86_64"}:
        if osname == "Darwin":
            return "macos-intel"
        if osname == "Windows":
            return "windows-amd64"
    elif m in {"aarch64", "arm64"}:
        if osname == "Darwin":
            return "macos-arm"
        if osname == "Windows":
            return "windows-arm"
    return f"{osname}-{machine}"


def fetch_latest_release() -> ReleaseInfo | None:
    current_platform = runtime_platform()

    if current_platform not in ASSET_PATTERNS:
        return None

    token = os.environ.get("ERLAB_GITHUB_TOKEN")
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": f"{REPO}-updater",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    r = requests.get(
        f"https://api.github.com/repos/{OWNER}/{REPO}/releases",
        headers=headers,
        timeout=20,
    )

    r.raise_for_status()
    releases = r.json()
    # Filter out drafts, maybe pre-releases
    filtered = [x for x in releases if not x.get("draft")]
    filtered = [x for x in filtered if not x.get("prerelease")]
    if not filtered:
        raise RuntimeError("No suitable releases found")
    # Sort by creation date descending
    filtered.sort(key=lambda x: x.get("created_at", ""), reverse=True)

    pat = ASSET_PATTERNS[current_platform]
    for rel in filtered:
        assets = rel.get("assets", [])
        chosen = None
        for a in assets:
            name = a.get("name", "")
            if pat.search(name or ""):
                chosen = ReleaseAsset(
                    name=name,
                    size=int(a.get("size", 0)),
                    download_url=a.get("browser_download_url"),
                    digest=a.get("digest", ""),
                )
                break
        if not chosen:
            continue

        return ReleaseInfo(
            tag=rel.get("tag_name", "0.0.0"),
            body=rel.get("body", ""),
            asset=chosen,
        )
    return None


def get_full_changelog_from(version: str) -> str:
    """Get changelog entries from releases newer than the given version."""
    token = os.environ.get("ERLAB_GITHUB_TOKEN")
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": f"{REPO}-updater",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    r = requests.get(
        f"https://api.github.com/repos/{OWNER}/{REPO}/releases",
        headers=headers,
        timeout=20,
    )

    r.raise_for_status()
    releases = r.json()
    # Filter out drafts, maybe pre-releases
    filtered = [x for x in releases if not x.get("draft")]
    filtered = [x for x in filtered if not x.get("prerelease")]
    if not filtered:
        raise RuntimeError("No suitable releases found")
    # Sort by creation date descending
    filtered.sort(key=lambda x: x.get("created_at", ""), reverse=True)

    changelog_entries = []
    for rel in filtered:
        tag = rel.get("tag_name", "v0.0.0")
        if tag == f"v{version}":
            break
        changelog_entries.append(rel.get("body", ""))

    return "\n\n".join(changelog_entries)


def verify_sha256(path: pathlib.Path, expected_hex: str) -> bool:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().lower() == expected_hex.removeprefix("sha256:").lower()


def get_install_root() -> pathlib.Path:
    """Return the root directory where the current app is installed.

    For macOS, this is the path to the .app bundle. For other platforms, the directory
    of the executable is returned.
    """
    if not erlab.utils.misc._IS_PACKAGED:
        return pathlib.Path("/Applications/ImageTool Manager.app").resolve()
    exe = pathlib.Path(sys.executable).resolve()
    if sys.platform == "darwin":
        bundle = exe.parents[2]
    else:
        bundle = exe.parent
    return bundle


def add_update_tmp_dir(tmpdir: pathlib.Path) -> None:
    """Add a temporary update directory to the QSettings for cleanup later."""
    settings = _get_updater_settings()
    tmp_dirs = settings.value("update_tmp_dirs", "")
    tmp_dir_list = tmp_dirs.strip().split(",") if tmp_dirs else []
    tmp_dirs_new = ",".join([*tmp_dir_list, str(tmpdir.resolve())])
    settings.setValue("update_tmp_dirs", tmp_dirs_new)
    settings.sync()
