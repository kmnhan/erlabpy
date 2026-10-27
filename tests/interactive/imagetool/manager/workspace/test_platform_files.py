"""Exercise workspace copies through platform-specific file APIs."""

from __future__ import annotations

import ctypes
import pathlib
import stat
import sys
import typing
from ctypes import wintypes

import h5py
import numpy as np
import pytest

import erlab.interactive.imagetool.manager._workspace._file_copy as workspace_file_copy
import erlab.interactive.imagetool.manager._workspace._storage as workspace_storage
import erlab.interactive.imagetool.manager._workspace._store as workspace_store
from erlab.interactive import _persistence_constants


@pytest.mark.skipif(sys.platform != "linux", reason="Requires Linux FICLONE")
def test_linux_clone_or_safe_decline(tmp_path):
    source = tmp_path / "source.itws"
    destination = tmp_path / "destination.itws"
    original = bytes(range(256)) * 1024
    source.write_bytes(original)
    copied = workspace_file_copy.copy_workspace_file(
        source, destination, allow_byte_copy=False
    )
    if copied:
        assert destination.read_bytes() == original
        destination.write_bytes(b"changed")
    else:
        assert not destination.exists()
    assert source.read_bytes() == original


@pytest.mark.skipif(sys.platform != "darwin", reason="Requires macOS clonefile")
def test_macos_clone_is_independent(tmp_path):
    source = tmp_path / "source α.itws"
    destination = tmp_path / "destination β.itws"
    original = bytes(range(256)) * 1024
    source.write_bytes(original)
    source.chmod(stat.S_IRUSR)
    try:
        assert workspace_file_copy.copy_workspace_file(
            source, destination, allow_byte_copy=False
        )
        assert destination.read_bytes() == original
        assert destination.stat().st_mode & stat.S_IWUSR
        assert not source.stat().st_mode & stat.S_IWUSR
        destination.write_bytes(b"changed")
        assert source.read_bytes() == original
    finally:
        source.chmod(stat.S_IRUSR | stat.S_IWUSR)


@pytest.mark.skipif(sys.platform != "win32", reason="Requires Windows CopyFileW")
def test_windows_copy_handles_long_unicode_paths(tmp_path):
    directory = pathlib.Path(workspace_file_copy._windows_path(tmp_path))
    for index in range(8):
        directory /= f"unicode αβγ directory {index:02d}"
    directory.mkdir(parents=True)
    source = directory / "source α.itws"
    destination = directory / "destination β.itws"
    assert len(str(source)) > 260
    original = bytes(range(256)) * 1024
    source.write_bytes(original)
    source.chmod(stat.S_IRUSR)
    try:
        assert workspace_file_copy._copy_native(
            source, destination, allow_byte_copy=True
        )
        assert destination.read_bytes() == original
        assert not destination.stat().st_mode & stat.S_IWUSR
        destination.chmod(stat.S_IRUSR | stat.S_IWUSR)
        destination.unlink()
        assert workspace_file_copy.copy_workspace_file(source, destination)
        assert destination.stat().st_mode & stat.S_IWUSR
        destination.write_bytes(b"changed")
        assert source.read_bytes() == original
        assert not source.stat().st_mode & stat.S_IWUSR
    finally:
        source.chmod(stat.S_IRUSR | stat.S_IWUSR)


@pytest.mark.skipif(sys.platform != "win32", reason="Requires Windows CopyFileW")
@pytest.mark.parametrize("attribute", [0x1, 0x2], ids=["readonly", "hidden"])
def test_windows_copy_preserves_raced_destination(monkeypatch, tmp_path, attribute):
    windows = typing.cast("typing.Any", ctypes)
    kernel32 = windows.WinDLL("kernel32", use_last_error=True)
    set_attributes = kernel32.SetFileAttributesW
    set_attributes.argtypes = [wintypes.LPCWSTR, wintypes.DWORD]
    set_attributes.restype = wintypes.BOOL
    source = tmp_path / "source.itws"
    destination = tmp_path / "destination.itws"
    source.write_bytes(b"source")
    native_copy = workspace_file_copy._copy_native

    def race_copy(src, dst, *, allow_byte_copy):
        dst.write_bytes(b"another owner")
        if not set_attributes(workspace_file_copy._windows_path(dst), attribute):
            raise windows.WinError(windows.get_last_error())
        return native_copy(src, dst, allow_byte_copy=allow_byte_copy)

    monkeypatch.setattr(workspace_file_copy, "_copy_native", race_copy)
    try:
        with pytest.raises(FileExistsError):
            workspace_file_copy.copy_workspace_file(source, destination)
        assert destination.read_bytes() == b"another owner"
        assert destination.stat().st_file_attributes & attribute
    finally:
        if destination.exists() and not set_attributes(
            workspace_file_copy._windows_path(destination), 0x80
        ):
            raise windows.WinError(windows.get_last_error())


@pytest.mark.skipif(sys.platform != "win32", reason="Requires Windows sharing modes")
def test_windows_copy_reports_sharing_violation(tmp_path):
    windows = typing.cast("typing.Any", ctypes)
    kernel32 = windows.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    create_file.restype = wintypes.HANDLE
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [wintypes.HANDLE]
    close_handle.restype = wintypes.BOOL
    source = tmp_path / "source.itws"
    destination = tmp_path / "destination.itws"
    source.write_bytes(b"source")
    # GENERIC_READ, no sharing, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL.
    handle = create_file(
        workspace_file_copy._windows_path(source),
        0x80000000,
        0,
        None,
        3,
        0x80,
        None,
    )
    if handle == ctypes.c_void_p(-1).value:
        raise windows.WinError(windows.get_last_error())
    try:
        with pytest.raises(PermissionError) as error:
            workspace_file_copy._copy_native(source, destination, allow_byte_copy=True)
        assert error.value.winerror == 32
        assert error.value.filename == str(source)
        assert error.value.filename2 == str(destination)
        assert not destination.exists()
    finally:
        if not close_handle(handle):
            raise windows.WinError(windows.get_last_error())


@pytest.mark.skipif(
    sys.platform not in {"darwin", "win32"},
    reason="Requires the macOS or Windows copy path",
)
@pytest.mark.parametrize("use_seed", [True, False], ids=["file-seed", "groups"])
def test_platform_workspace_copy_survives_source_removal(tmp_path, use_seed):
    source = tmp_path / "source α.itws"
    target = tmp_path / "destination β.itws"
    object_path = workspace_store.WorkspaceStore.object_path("kept")
    manifest = {
        "schema_version": _persistence_constants.WORKSPACE_SCHEMA_VERSION,
        "nodes": [
            {
                "uid": "image",
                "path": "0",
                "kind": "imagetool",
                "payload_object_id": "kept",
                "payload_path": object_path,
            }
        ],
        "root_order": ["image"],
    }
    bits = np.array(
        [
            0x7FF8000000000042,
            0x8000000000000000,
            0x3FF0000000000000,
            0xBFF0000000000000,
            0x7FF0000000000000,
            0x0000000000000001,
        ],
        dtype=np.uint64,
    ).reshape(2, 3)
    with workspace_store.WorkspaceStore(source, create=True) as source_store:
        with source_store.write_session() as source_h5:
            source_h5.require_group(object_path).create_dataset(
                "data", data=bits.view(np.float64), compression="gzip"
            )
        source_store.publish(manifest)
        source_id = source_store.workspace_id
        plan = workspace_storage._WorkspaceGenerationPlan(
            manifest=manifest,
            objects=(
                workspace_storage._WorkspaceObjectWrite(
                    "kept", source_file=str(source), source_path=object_path
                ),
            ),
        )
        with source_store.read_session() as source_h5:
            assert source_h5.mode == "r"
            if use_seed:
                assert workspace_storage._seed_workspace_save_as(target, plan)
            with workspace_store.WorkspaceStore(
                target, create=not use_seed
            ) as target_store:
                workspace_storage._write_workspace_generation(
                    target_store, plan, compression_mode="none"
                )
                assert target_store.workspace_id != source_id
                assert target_store.current_generation().manifest["root_order"] == [
                    "image"
                ]
            np.testing.assert_array_equal(
                source_h5[object_path]["data"][()].view(np.uint64), bits
            )
    source.unlink()
    with h5py.File(target, "r+") as target_h5:
        np.testing.assert_array_equal(
            target_h5[object_path]["data"][()].view(np.uint64), bits
        )
        target_h5[object_path]["data"][0, 0] = 42.0
    with h5py.File(target, "r") as target_h5:
        assert target_h5[object_path]["data"][0, 0] == 42.0
