"""Utilities that don't fit in any other category."""

__all__ = [
    "emit_user_level_warning",
    "get_tqdm",
    "is_interactive",
    "is_newer_version",
    "is_sequence_of",
    "open_in_file_manager",
]

import inspect
import os
import pathlib
import subprocess
import sys
import typing
import warnings
from collections.abc import Sequence

import numpy as np

if typing.TYPE_CHECKING:
    import tqdm

_IS_PACKAGED: bool = getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")


_NestedGeneric = np.generic | list["_NestedGeneric"] | typing.Any


@typing.overload
def _convert_to_native(obj: np.generic) -> typing.Any: ...
@typing.overload
def _convert_to_native(obj: list[_NestedGeneric]) -> list[typing.Any]: ...
@typing.overload
def _convert_to_native(obj: typing.Any) -> typing.Any: ...
def _convert_to_native(obj: _NestedGeneric) -> typing.Any:
    """Convert numpy objects to native types."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, list):
        return [_convert_to_native(item) for item in obj]
    return obj


def _find_stack_level() -> int:
    """Find the first place in the stack that is not inside erlab, xarray, or stdlib.

    This is unless the code emanates from a test, in which case we would prefer to see
    the source.

    This function is adapted from xarray.core.utils.find_stack_level.

    Returns
    -------
    stacklevel : int
        First level in the stack that is not part of erlab or stdlib.
    """
    import xarray

    import erlab

    xarray_dir = pathlib.Path(xarray.__file__).parent
    pkg_dir = pathlib.Path(erlab.__file__).parent.parent.parent
    test_dir = pkg_dir / "tests"

    std_lib_init = sys.modules["os"].__file__
    if std_lib_init is None:
        return 0

    std_lib_dir = pathlib.Path(std_lib_init).parent

    frame = inspect.currentframe()
    n = 0
    while frame:
        fname = inspect.getfile(frame)
        if (
            (fname.startswith(str(pkg_dir)) and not fname.startswith(str(test_dir)))
            or (
                fname.startswith(str(std_lib_dir))
                and "site-packages" not in fname
                and "dist-packages" not in fname
            )
            or fname.startswith(str(xarray_dir))
        ):
            frame = frame.f_back
            n += 1
        else:
            break
    return n


def emit_user_level_warning(message, category=None) -> None:
    """Emit a warning at the user level by inspecting the stack trace."""
    stacklevel = _find_stack_level()
    return warnings.warn(message, category=category, stacklevel=stacklevel)


_T = typing.TypeVar("_T")


def is_sequence_of(
    val: typing.Any, element_type: type[_T]
) -> typing.TypeGuard[Sequence[_T]]:
    """
    Check if the given object is a sequence of elements of the specified type.

    Parameters
    ----------
    val
        The object to check.
    element_type
        The type of elements that the sequence should contain.

    Returns
    -------
    bool
        `True` if `val` is a sequence and all elements in the sequence are of type
        `element_type`, `False` otherwise.

    Examples
    --------
    >>> is_sequence_of([1, 2, 3], int)
    True

    >>> is_sequence_of([1, 2, "3"], int)
    False
    """
    return isinstance(val, Sequence) and all(isinstance(x, element_type) for x in val)


def is_interactive() -> bool:
    """Check if the code is running in an interactive environment.

    Returns
    -------
    bool
        `True` if the code is running in an interactive environment (IPython), `False`
        otherwise.
    """
    try:
        shell = get_ipython().__class__.__name__  # type: ignore[name-defined]
        if shell in ["ZMQInteractiveShell", "TerminalInteractiveShell"]:
            return True
    except NameError:
        pass
    return False


def open_in_file_manager(path: str | os.PathLike) -> None:  # pragma: no cover
    """Reveal a path in the system's file manager.

    Parameters
    ----------
    path
        Path to the file or folder.
    """
    path = pathlib.Path(path).resolve()
    if path.is_dir():
        if sys.platform == "win32":
            os.startfile(path)  # noqa: S606
        else:
            open_cmd = "open" if sys.platform == "darwin" else "xdg-open"
            subprocess.call([open_cmd, str(path)])
    else:
        if sys.platform == "darwin":
            subprocess.call(["open", "-R", str(path)])
        elif sys.platform == "win32":
            subprocess.call(["explorer", "/select,", str(path)])
        else:
            # We can't do this reliably on Linux, so we just open the folder
            subprocess.call(["xdg-open", str(path.parent)])


def is_newer_version(version_str: str) -> bool:  # pragma: no cover
    """Check if a version string is newer than the current installed erlab version."""
    import packaging.version

    import erlab

    return packaging.version.Version(version_str) > packaging.version.Version(
        erlab.__version__
    )


def get_tqdm() -> type["tqdm.tqdm"]:
    """Get the appropriate tqdm module.

    For frozen packages, we cannot use `tqdm.auto` since it causes issues with
    incomplete IPython installations.
    """
    if _IS_PACKAGED:
        import tqdm
    else:
        import tqdm.auto as tqdm

    return tqdm.tqdm
