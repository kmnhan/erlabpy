"""Native file copying with a portable fallback for workspace saves."""

from __future__ import annotations

import ctypes
import errno
import ntpath
import os
import pathlib
import shutil
import stat
import sys
import typing

_UNSUPPORTED_CLONE_ERRNOS = frozenset(
    {
        errno.EXDEV,
        errno.EINVAL,
        # FICLONE can report EBADF for a filesystem without reflink support.
        errno.EBADF,
        errno.ENOSYS,
        getattr(errno, "ENOTSUP", errno.ENOSYS),
        getattr(errno, "EOPNOTSUPP", errno.ENOSYS),
        errno.ENOTTY,
    }
)
# ERROR_INVALID_FUNCTION, ERROR_NOT_SUPPORTED, ERROR_CALL_NOT_IMPLEMENTED.
_UNSUPPORTED_WINDOWS_ERRORS = frozenset({1, 50, 120})


def _windows_path(path: pathlib.Path) -> str:
    value = ntpath.abspath(os.fspath(path))
    if value.startswith(("\\\\?\\", "\\\\.\\")):
        return value
    if value.startswith("\\\\"):
        return "\\\\?\\UNC\\" + value[2:]
    return "\\\\?\\" + value


def _require_absent_destination(destination: pathlib.Path) -> None:
    if os.path.lexists(destination):
        raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), destination)


def _copy_native(
    source: pathlib.Path, destination: pathlib.Path, *, allow_byte_copy: bool
) -> bool:
    if sys.platform == "darwin":
        library = ctypes.CDLL(None, use_errno=True)
        clonefile = getattr(library, "clonefile", None)
        if clonefile is None:
            return False
        clonefile.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
        clonefile.restype = ctypes.c_int
        if clonefile(os.fsencode(source), os.fsencode(destination), 0):
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code), os.fspath(destination))
        return True

    if sys.platform == "linux":
        import fcntl

        # Python 3.11 does not expose FICLONE. Its regular copy still uses
        # shutil's kernel-assisted copy where available.
        request = getattr(fcntl, "FICLONE", None)
        if request is None:
            return False
        try:
            src = source.open("rb")
        except OSError:
            _require_absent_destination(destination)
            raise
        with src, destination.open("xb") as dst:
            fcntl.ioctl(dst.fileno(), request, src.fileno())
        return True

    if sys.platform == "win32" and allow_byte_copy:
        windows = typing.cast("typing.Any", ctypes)
        copy_file = windows.WinDLL("kernel32", use_last_error=True).CopyFileW
        copy_file.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_int]
        copy_file.restype = ctypes.c_int
        if not copy_file(_windows_path(source), _windows_path(destination), 1):
            error = windows.WinError(windows.get_last_error())
            error.filename = os.fspath(source)
            error.filename2 = os.fspath(destination)
            if error.winerror == 5:
                # An existing hidden or read-only file can report access denied.
                # Its ownership is unknown, so do not remove it as a partial copy.
                _require_absent_destination(destination)
            raise error
        return True

    return False


def _remove_incomplete_copy(destination: pathlib.Path) -> None:
    try:
        mode = destination.stat().st_mode
    except FileNotFoundError:
        return
    # CopyFileW can leave a read-only file. Remove only our new temporary file.
    if not mode & stat.S_IWUSR:
        destination.chmod(mode | stat.S_IWUSR)
    destination.unlink(missing_ok=True)


def copy_workspace_file(
    source: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    *,
    allow_byte_copy: bool = True,
) -> bool:
    """Create a writable independent copy at a previously absent path.

    Return False when byte copying is disabled and cloning is unavailable.
    """
    source = pathlib.Path(source)
    destination = pathlib.Path(destination)
    _require_absent_destination(destination)
    try:
        source_mode = source.stat().st_mode
    except OSError:
        _require_absent_destination(destination)
        raise
    if not stat.S_ISREG(source_mode):
        _require_absent_destination(destination)
        raise OSError(errno.EINVAL, "Workspace source must be a regular file", source)

    copied = False
    try:
        try:
            copied = _copy_native(source, destination, allow_byte_copy=allow_byte_copy)
        except FileExistsError:
            # A different owner created the destination after the initial check.
            raise
        except OSError as error:
            _remove_incomplete_copy(destination)
            windows_error = getattr(error, "winerror", None)
            unsupported = (
                windows_error in _UNSUPPORTED_WINDOWS_ERRORS
                if windows_error is not None
                else error.errno in _UNSUPPORTED_CLONE_ERRNOS
            )
            if not unsupported:
                raise

        if not copied and allow_byte_copy:
            # Reserve the destination exclusively before shutil opens it for its
            # platform's fast copy. Never overwrite an existing Save As target.
            with destination.open("xb"):
                pass
            shutil.copyfile(source, destination)
            copied = True
        if copied:
            mode = destination.stat().st_mode
            if not mode & stat.S_IWUSR:
                destination.chmod(mode | stat.S_IWUSR)
    except FileExistsError:
        raise
    except BaseException:
        _remove_incomplete_copy(destination)
        raise
    if not copied:
        _require_absent_destination(destination)
    return copied
