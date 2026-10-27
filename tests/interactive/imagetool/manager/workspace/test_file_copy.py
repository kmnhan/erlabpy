from __future__ import annotations

import ctypes
import errno
import os
import pathlib
import shutil
import stat
import types

import pytest

import erlab.interactive.imagetool.manager._workspace._file_copy as workspace_file_copy


@pytest.fixture
def copy_paths(tmp_path):
    source = tmp_path / "source α.itws"
    destination = tmp_path / "destination β.itws"
    source.write_bytes(bytes(range(256)) * 1024)
    return source, destination


def test_copy_workspace_file_creates_independent_writable_file(copy_paths):
    source, destination = copy_paths
    source.chmod(stat.S_IRUSR)
    original = source.read_bytes()
    try:
        assert workspace_file_copy.copy_workspace_file(source, destination)
        assert destination.read_bytes() == original
        assert destination.stat().st_mode & stat.S_IWUSR
        assert not source.stat().st_mode & stat.S_IWUSR
        destination.write_bytes(b"changed")
        assert source.read_bytes() == original
    finally:
        source.chmod(stat.S_IRUSR | stat.S_IWUSR)


def test_copy_workspace_file_portable_fallback(monkeypatch, copy_paths):
    source, destination = copy_paths
    monkeypatch.setattr(workspace_file_copy, "_copy_native", lambda *_, **__: False)
    original_copyfile = shutil.copyfile
    calls = []

    def copyfile(src, dst):
        assert dst.exists()
        assert dst.stat().st_size == 0
        calls.append((src, dst))
        return original_copyfile(src, dst)

    monkeypatch.setattr(workspace_file_copy.shutil, "copyfile", copyfile)
    assert workspace_file_copy.copy_workspace_file(source, destination)
    assert calls == [(source, destination)]
    assert destination.read_bytes() == source.read_bytes()


@pytest.mark.parametrize("allow_byte_copy", [False, True])
def test_copy_workspace_file_preserves_existing_destination(
    monkeypatch, copy_paths, allow_byte_copy
):
    source, destination = copy_paths
    destination.write_bytes(b"existing")
    monkeypatch.setattr(
        workspace_file_copy,
        "_copy_native",
        lambda *_, **__: pytest.fail("Copying must not start"),
    )
    with pytest.raises(FileExistsError):
        workspace_file_copy.copy_workspace_file(
            source, destination, allow_byte_copy=allow_byte_copy
        )
    assert destination.read_bytes() == b"existing"


@pytest.mark.parametrize("allow_byte_copy", [False, True])
def test_copy_workspace_file_preserves_raced_destination(
    monkeypatch, copy_paths, allow_byte_copy
):
    source, destination = copy_paths

    def native(*_, **__):
        destination.write_bytes(b"another owner")
        raise FileExistsError(errno.EEXIST, "already created", destination)

    monkeypatch.setattr(workspace_file_copy, "_copy_native", native)
    with pytest.raises(FileExistsError):
        workspace_file_copy.copy_workspace_file(
            source, destination, allow_byte_copy=allow_byte_copy
        )
    assert destination.read_bytes() == b"another owner"


def test_copy_workspace_file_rejects_non_file(tmp_path):
    with pytest.raises(OSError, match="regular file"):
        workspace_file_copy.copy_workspace_file(tmp_path, tmp_path / "destination")


def test_copy_workspace_file_missing_source(tmp_path):
    source = tmp_path / "missing.itws"
    destination = tmp_path / "destination.itws"
    with pytest.raises(FileNotFoundError):
        workspace_file_copy.copy_workspace_file(source, destination)
    assert not destination.exists()


@pytest.mark.parametrize("code", [errno.ENOENT, errno.EACCES])
def test_copy_workspace_file_source_stat_failure_preserves_raced_destination(
    monkeypatch, copy_paths, code
):
    source, destination = copy_paths
    original_stat = pathlib.Path.stat

    def stat_source(path, *args, **kwargs):
        if path == source:
            destination.write_bytes(b"another owner")
            raise OSError(code, "Cannot inspect source", source)
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "stat", stat_source)
    with pytest.raises(FileExistsError):
        workspace_file_copy.copy_workspace_file(source, destination)
    assert destination.read_bytes() == b"another owner"


def test_copy_workspace_file_removes_partial_portable_copy(monkeypatch, copy_paths):
    source, destination = copy_paths
    monkeypatch.setattr(workspace_file_copy, "_copy_native", lambda *_, **__: False)

    def fail_copy(src, dst):
        dst.write_bytes(b"partial")
        dst.chmod(stat.S_IRUSR)
        raise OSError(errno.ENOSPC, "disk full")

    monkeypatch.setattr(workspace_file_copy.shutil, "copyfile", fail_copy)
    with pytest.raises(OSError, match="disk full") as error:
        workspace_file_copy.copy_workspace_file(source, destination)
    assert error.value.errno == errno.ENOSPC
    assert not destination.exists()


@pytest.mark.parametrize("allow_byte_copy", [False, True])
def test_copy_workspace_file_fallback_reserves_destination(
    monkeypatch, copy_paths, allow_byte_copy
):
    source, destination = copy_paths

    def native(*_, **__):
        destination.write_bytes(b"another owner")
        return False

    monkeypatch.setattr(workspace_file_copy, "_copy_native", native)
    with pytest.raises(FileExistsError):
        workspace_file_copy.copy_workspace_file(
            source, destination, allow_byte_copy=allow_byte_copy
        )
    assert destination.read_bytes() == b"another owner"


def test_copy_workspace_file_cleanup_failure_is_not_hidden(monkeypatch, copy_paths):
    source, destination = copy_paths

    def fail_native(*_, **__):
        raise OSError(errno.ENOSPC, "disk full")

    def fail_cleanup(*_):
        raise PermissionError(errno.EACCES, "cannot remove temporary copy")

    monkeypatch.setattr(workspace_file_copy, "_copy_native", fail_native)
    monkeypatch.setattr(workspace_file_copy, "_remove_incomplete_copy", fail_cleanup)
    with pytest.raises(PermissionError, match="cannot remove"):
        workspace_file_copy.copy_workspace_file(source, destination)


def _mock_macos_clone(monkeypatch, callback):
    class Clone:
        def __call__(self, source, destination, flags):
            return callback(source, destination, flags)

    clone = Clone()
    monkeypatch.setattr(workspace_file_copy.sys, "platform", "darwin")
    monkeypatch.setattr(
        workspace_file_copy.ctypes,
        "CDLL",
        lambda name, *, use_errno: types.SimpleNamespace(clonefile=clone),
    )
    return clone


@pytest.mark.parametrize("allow_byte_copy", [False, True])
def test_copy_workspace_file_macos_clone(monkeypatch, copy_paths, allow_byte_copy):
    source, destination = copy_paths

    def callback(src, dst, flags):
        assert (src, dst, flags) == (os.fsencode(source), os.fsencode(destination), 0)
        destination.write_bytes(source.read_bytes())
        destination.chmod(stat.S_IRUSR)
        return 0

    clone = _mock_macos_clone(monkeypatch, callback)
    assert workspace_file_copy.copy_workspace_file(
        source, destination, allow_byte_copy=allow_byte_copy
    )
    assert clone.argtypes == [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
    assert clone.restype == ctypes.c_int
    assert destination.read_bytes() == source.read_bytes()
    assert destination.stat().st_mode & stat.S_IWUSR


@pytest.mark.parametrize("allow_byte_copy", [False, True])
def test_copy_workspace_file_missing_macos_clone(
    monkeypatch, copy_paths, allow_byte_copy
):
    source, destination = copy_paths
    monkeypatch.setattr(workspace_file_copy.sys, "platform", "darwin")
    monkeypatch.setattr(
        workspace_file_copy.ctypes, "CDLL", lambda *_, **__: types.SimpleNamespace()
    )
    if not allow_byte_copy:
        monkeypatch.setattr(
            workspace_file_copy.shutil,
            "copyfile",
            lambda *_: pytest.fail("Byte copying must not start"),
        )
    assert (
        workspace_file_copy.copy_workspace_file(
            source, destination, allow_byte_copy=allow_byte_copy
        )
        is allow_byte_copy
    )
    if allow_byte_copy:
        assert destination.read_bytes() == source.read_bytes()
    else:
        assert not destination.exists()


@pytest.mark.parametrize(
    "code",
    [errno.EXDEV, getattr(errno, "ENOTSUP", errno.ENOSYS), errno.ENOSYS, errno.EINVAL],
)
@pytest.mark.parametrize("allow_byte_copy", [False, True])
def test_copy_workspace_file_unsupported_clone_falls_back(
    monkeypatch, copy_paths, code, allow_byte_copy
):
    source, destination = copy_paths

    def callback(*_):
        destination.write_bytes(b"partial")
        destination.chmod(stat.S_IRUSR)
        ctypes.set_errno(code)
        return -1

    _mock_macos_clone(monkeypatch, callback)
    if not allow_byte_copy:
        monkeypatch.setattr(
            workspace_file_copy.shutil,
            "copyfile",
            lambda *_: pytest.fail("Byte copying must not start"),
        )
    assert (
        workspace_file_copy.copy_workspace_file(
            source, destination, allow_byte_copy=allow_byte_copy
        )
        is allow_byte_copy
    )
    if allow_byte_copy:
        assert destination.read_bytes() == source.read_bytes()
    else:
        assert not destination.exists()


@pytest.mark.parametrize("code", [errno.EACCES, errno.ENOSPC, errno.EIO])
@pytest.mark.parametrize("allow_byte_copy", [False, True])
def test_copy_workspace_file_real_clone_errors_propagate(
    monkeypatch, copy_paths, code, allow_byte_copy
):
    source, destination = copy_paths

    def callback(*_):
        destination.write_bytes(b"partial")
        ctypes.set_errno(code)
        return -1

    _mock_macos_clone(monkeypatch, callback)
    with pytest.raises(OSError, match=rf"\[Errno {code}\]") as error:
        workspace_file_copy.copy_workspace_file(
            source, destination, allow_byte_copy=allow_byte_copy
        )
    assert error.value.errno == code
    assert not destination.exists()


@pytest.mark.parametrize("allow_byte_copy", [False, True])
def test_copy_workspace_file_linux_reflink(monkeypatch, copy_paths, allow_byte_copy):
    fcntl = pytest.importorskip("fcntl")

    source, destination = copy_paths
    request = 123456
    calls = []

    def ioctl(dst_fd, command, src_fd):
        assert command == request
        calls.append((dst_fd, src_fd))
        os.write(dst_fd, os.read(src_fd, source.stat().st_size))
        return 0

    monkeypatch.setattr(workspace_file_copy.sys, "platform", "linux")
    monkeypatch.setattr(fcntl, "FICLONE", request, raising=False)
    monkeypatch.setattr(fcntl, "ioctl", ioctl)
    assert workspace_file_copy.copy_workspace_file(
        source, destination, allow_byte_copy=allow_byte_copy
    )
    assert len(calls) == 1
    assert destination.read_bytes() == source.read_bytes()


@pytest.mark.parametrize("code", sorted(workspace_file_copy._UNSUPPORTED_CLONE_ERRNOS))
@pytest.mark.parametrize("allow_byte_copy", [False, True])
def test_copy_workspace_file_linux_unsupported_reflink(
    monkeypatch, copy_paths, code, allow_byte_copy
):
    fcntl = pytest.importorskip("fcntl")

    source, destination = copy_paths
    file_descriptors = []

    def ioctl(dst_fd, _request, src_fd):
        file_descriptors.extend((src_fd, dst_fd))
        os.write(dst_fd, b"partial")
        raise OSError(code, "no reflinks")

    monkeypatch.setattr(workspace_file_copy.sys, "platform", "linux")
    monkeypatch.setattr(fcntl, "FICLONE", 123456, raising=False)
    monkeypatch.setattr(fcntl, "ioctl", ioctl)
    if not allow_byte_copy:
        monkeypatch.setattr(
            workspace_file_copy.shutil,
            "copyfile",
            lambda *_: pytest.fail("Byte copying must not start"),
        )
    assert (
        workspace_file_copy.copy_workspace_file(
            source, destination, allow_byte_copy=allow_byte_copy
        )
        is allow_byte_copy
    )
    if allow_byte_copy:
        assert destination.read_bytes() == source.read_bytes()
    else:
        assert not destination.exists()
    for descriptor in file_descriptors:
        with pytest.raises(OSError, match=rf"\[Errno {errno.EBADF}\]") as error:
            os.fstat(descriptor)
        assert error.value.errno == errno.EBADF


@pytest.mark.parametrize("code", [errno.EACCES, errno.ENOSPC, errno.EIO])
@pytest.mark.parametrize("allow_byte_copy", [False, True])
def test_copy_workspace_file_linux_reflink_failure_closes_files(
    monkeypatch, copy_paths, code, allow_byte_copy
):
    fcntl = pytest.importorskip("fcntl")
    source, destination = copy_paths
    file_descriptors = []

    def ioctl(dst_fd, _request, src_fd):
        file_descriptors.extend((src_fd, dst_fd))
        os.write(dst_fd, b"partial")
        raise OSError(code, "Cannot clone source")

    monkeypatch.setattr(workspace_file_copy.sys, "platform", "linux")
    monkeypatch.setattr(fcntl, "FICLONE", 123456, raising=False)
    monkeypatch.setattr(fcntl, "ioctl", ioctl)
    with pytest.raises(OSError, match="Cannot clone source") as error:
        workspace_file_copy.copy_workspace_file(
            source, destination, allow_byte_copy=allow_byte_copy
        )
    assert error.value.errno == code
    assert not destination.exists()
    for descriptor in file_descriptors:
        with pytest.raises(OSError, match=rf"\[Errno {errno.EBADF}\]") as error:
            os.fstat(descriptor)
        assert error.value.errno == errno.EBADF


@pytest.mark.parametrize("code", [errno.ENOENT, errno.EACCES])
@pytest.mark.parametrize("raced_destination", [False, True])
def test_copy_workspace_file_linux_source_open_preserves_raced_destination(
    monkeypatch, copy_paths, code, raced_destination
):
    fcntl = pytest.importorskip("fcntl")
    source, destination = copy_paths
    original_open = pathlib.Path.open

    def open_source(path, mode="r", *args, **kwargs):
        if path == source and mode == "rb":
            if raced_destination:
                destination.write_bytes(b"another owner")
            raise OSError(code, "Cannot open source", source)
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(workspace_file_copy.sys, "platform", "linux")
    monkeypatch.setattr(fcntl, "FICLONE", 123456, raising=False)
    monkeypatch.setattr(pathlib.Path, "open", open_source)
    if raced_destination:
        with pytest.raises(FileExistsError):
            workspace_file_copy.copy_workspace_file(source, destination)
        assert destination.read_bytes() == b"another owner"
    else:
        with pytest.raises(OSError, match="Cannot open source") as error:
            workspace_file_copy.copy_workspace_file(source, destination)
        assert error.value.errno == code
        assert not destination.exists()


@pytest.mark.parametrize("allow_byte_copy", [False, True])
def test_copy_workspace_file_linux_without_ficlone(
    monkeypatch, copy_paths, allow_byte_copy
):
    fcntl = pytest.importorskip("fcntl")

    source, destination = copy_paths
    monkeypatch.setattr(workspace_file_copy.sys, "platform", "linux")
    monkeypatch.delattr(fcntl, "FICLONE", raising=False)
    if not allow_byte_copy:
        monkeypatch.setattr(
            workspace_file_copy.shutil,
            "copyfile",
            lambda *_: pytest.fail("Byte copying must not start"),
        )
    assert (
        workspace_file_copy.copy_workspace_file(
            source, destination, allow_byte_copy=allow_byte_copy
        )
        is allow_byte_copy
    )
    if allow_byte_copy:
        assert destination.read_bytes() == source.read_bytes()
    else:
        assert not destination.exists()


def _mock_windows_copy(monkeypatch, callback, *, error_code=0):
    native_win_error = getattr(ctypes, "WinError", None)

    class Copy:
        def __call__(self, source, destination, fail_if_exists):
            return callback(source, destination, fail_if_exists)

    copy = Copy()
    monkeypatch.setattr(workspace_file_copy.sys, "platform", "win32")
    monkeypatch.setattr(
        ctypes,
        "WinDLL",
        lambda name, *, use_last_error: types.SimpleNamespace(CopyFileW=copy),
        raising=False,
    )
    monkeypatch.setattr(ctypes, "get_last_error", lambda: error_code, raising=False)

    def win_error(code):
        if native_win_error is not None:
            return native_win_error(code)
        mapped = {
            2: errno.ENOENT,
            3: errno.ENOENT,
            5: errno.EACCES,
            32: errno.EACCES,
            33: errno.EACCES,
            80: errno.EEXIST,
            112: errno.ENOSPC,
            183: errno.EEXIST,
        }
        error = OSError(mapped.get(code, errno.EINVAL), "Windows copy failed")
        error.winerror = code
        return error

    monkeypatch.setattr(ctypes, "WinError", win_error, raising=False)
    return copy


def test_copy_workspace_file_windows_copy(monkeypatch, copy_paths):
    source, destination = copy_paths

    def callback(src, dst, fail_if_exists):
        assert src == workspace_file_copy._windows_path(source)
        assert dst == workspace_file_copy._windows_path(destination)
        assert fail_if_exists == 1
        destination.write_bytes(source.read_bytes())
        destination.chmod(stat.S_IRUSR)
        return 1

    copy = _mock_windows_copy(monkeypatch, callback)
    assert workspace_file_copy.copy_workspace_file(source, destination)
    assert copy.argtypes == [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_int]
    assert copy.restype == ctypes.c_int
    assert destination.read_bytes() == source.read_bytes()
    assert destination.stat().st_mode & stat.S_IWUSR


def test_copy_workspace_file_windows_clone_only_skips_byte_copies(
    monkeypatch, copy_paths
):
    source, destination = copy_paths
    _mock_windows_copy(
        monkeypatch, lambda *_: pytest.fail("Windows byte copying must not start")
    )
    monkeypatch.setattr(
        workspace_file_copy.shutil,
        "copyfile",
        lambda *_: pytest.fail("Portable byte copying must not start"),
    )
    assert not workspace_file_copy.copy_workspace_file(
        source, destination, allow_byte_copy=False
    )
    assert not destination.exists()


@pytest.mark.parametrize("code", [1, 50, 120])
def test_copy_workspace_file_windows_unsupported_copy(monkeypatch, copy_paths, code):
    source, destination = copy_paths

    def callback(*_):
        destination.write_bytes(b"partial")
        return 0

    _mock_windows_copy(monkeypatch, callback, error_code=code)
    workspace_file_copy.copy_workspace_file(source, destination)
    assert destination.read_bytes() == source.read_bytes()


@pytest.mark.parametrize("code", [5, 32, 33])
def test_copy_workspace_file_windows_access_error(monkeypatch, copy_paths, code):
    source, destination = copy_paths
    _mock_windows_copy(monkeypatch, lambda *_: 0, error_code=code)
    with pytest.raises(PermissionError) as error:
        workspace_file_copy.copy_workspace_file(source, destination)
    assert error.value.winerror == code
    assert error.value.errno == errno.EACCES
    assert error.value.filename == str(source)
    assert error.value.filename2 == str(destination)
    assert not destination.exists()


@pytest.mark.parametrize("code", [80, 183])
def test_copy_workspace_file_windows_raced_destination(monkeypatch, copy_paths, code):
    source, destination = copy_paths

    def callback(*_):
        destination.write_bytes(b"another owner")
        return 0

    _mock_windows_copy(monkeypatch, callback, error_code=code)
    with pytest.raises(FileExistsError):
        workspace_file_copy.copy_workspace_file(source, destination)
    assert destination.read_bytes() == b"another owner"


def test_copy_workspace_file_windows_readonly_raced_destination(
    monkeypatch, copy_paths
):
    source, destination = copy_paths

    def callback(*_):
        destination.write_bytes(b"another owner")
        destination.chmod(stat.S_IRUSR)
        return 0

    _mock_windows_copy(monkeypatch, callback, error_code=5)
    try:
        with pytest.raises(FileExistsError):
            workspace_file_copy.copy_workspace_file(source, destination)
        assert destination.read_bytes() == b"another owner"
        assert not destination.stat().st_mode & stat.S_IWUSR
    finally:
        destination.chmod(stat.S_IRUSR | stat.S_IWUSR)


@pytest.mark.skipif(os.name == "nt", reason="Requires POSIX symlink creation")
def test_copy_workspace_file_windows_preserves_raced_dangling_symlink(
    monkeypatch, copy_paths
):
    source, destination = copy_paths
    missing = source.with_name("missing.itws")

    def callback(*_):
        destination.symlink_to(missing)
        return 0

    _mock_windows_copy(monkeypatch, callback, error_code=5)
    with pytest.raises(FileExistsError):
        workspace_file_copy.copy_workspace_file(source, destination)
    assert destination.is_symlink()
    assert destination.readlink() == missing


def test_copy_workspace_file_windows_full_disk_removes_partial_copy(
    monkeypatch, copy_paths
):
    source, destination = copy_paths

    def callback(*_):
        destination.write_bytes(b"partial")
        destination.chmod(stat.S_IRUSR)
        return 0

    _mock_windows_copy(monkeypatch, callback, error_code=112)
    with pytest.raises(OSError, match=rf"WinError 112|Errno {errno.ENOSPC}") as error:
        workspace_file_copy.copy_workspace_file(source, destination)
    assert error.value.winerror == 112
    assert error.value.errno == errno.ENOSPC
    assert not destination.exists()


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (r"C:\folder\α.itws", r"\\?\C:\folder\α.itws"),
        (r"\\server\folder\α.itws", r"\\?\UNC\server\folder\α.itws"),
        (r"\\?\C:\folder\α.itws", r"\\?\C:\folder\α.itws"),
        (r"\\.\C:\folder\α.itws", r"\\.\C:\folder\α.itws"),
    ],
)
def test_workspace_copy_windows_paths(value, expected):
    assert workspace_file_copy._windows_path(pathlib.PureWindowsPath(value)) == expected


@pytest.mark.parametrize("allow_byte_copy", [False, True])
def test_copy_workspace_file_unknown_platform(monkeypatch, copy_paths, allow_byte_copy):
    source, destination = copy_paths
    monkeypatch.setattr(workspace_file_copy.sys, "platform", "other")
    if not allow_byte_copy:
        monkeypatch.setattr(
            workspace_file_copy.shutil,
            "copyfile",
            lambda *_: pytest.fail("Byte copying must not start"),
        )
    assert (
        workspace_file_copy.copy_workspace_file(
            source, destination, allow_byte_copy=allow_byte_copy
        )
        is allow_byte_copy
    )
    if allow_byte_copy:
        assert destination.read_bytes() == source.read_bytes()
    else:
        assert not destination.exists()
