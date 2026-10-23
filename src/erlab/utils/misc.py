"""Utilities that don't fit in any other category."""

__all__ = [
    "accepts_kwarg",
    "emit_user_level_warning",
    "get_tqdm",
    "is_interactive",
    "is_newer_version",
    "is_sequence_of",
    "open_in_file_manager",
]

import functools
import inspect
import keyword
import os
import pathlib
import subprocess
import sys
import typing
import unicodedata
import warnings
from collections.abc import Callable, Sequence

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


@functools.cache
def _warning_skip_file_prefixes() -> tuple[str, ...]:
    """Locate package and standard-library frames without skipping site packages."""
    import xarray

    prefixes = [
        str(pathlib.Path(__file__).parents[1]) + os.sep,
        str(pathlib.Path(xarray.__file__).parent) + os.sep,
    ]
    if os.__file__ is not None:
        stdlib_dir = pathlib.Path(os.__file__).parent
        if stdlib_dir.is_dir():
            prefixes.extend(
                str(path) + (os.sep if path.is_dir() else "")
                for path in stdlib_dir.iterdir()
                if path.name not in {"site-packages", "dist-packages"}
            )
    return tuple(prefixes)


def emit_user_level_warning(
    message: str | Warning, category: type[Warning] | None = None
) -> None:
    """Emit a warning at the first caller outside ERLabPy, xarray, and the stdlib."""
    warnings.warn(
        message, category=category, skip_file_prefixes=_warning_skip_file_prefixes()
    )


def _is_valid_identifier(value: typing.Any) -> bool:
    return (
        isinstance(value, str) and value.isidentifier() and not keyword.iskeyword(value)
    )


def _normalize_identifier(value: str | None) -> str:
    """Return a valid Python identifier derived from arbitrary user text."""
    text = unicodedata.normalize("NFKC", "" if value is None else value.strip())
    normalized_chars: list[str] = []
    for index, char in enumerate(text):
        if (index == 0 and char.isidentifier()) or (
            index > 0 and f"a{char}".isidentifier()
        ):
            normalized_chars.append(char)
        elif index == 0 and f"a{char}".isidentifier():
            normalized_chars.extend(("_", char))
        else:
            normalized_chars.append("_")
    normalized = "".join(normalized_chars)
    if not normalized or not normalized.strip("_"):
        normalized = "var"
    if keyword.iskeyword(normalized):
        normalized += "_"
    if not _is_valid_identifier(normalized):
        raise RuntimeError(f"Failed to normalize {value!r} as a Python identifier")
    return normalized


def is_sequence_of[T](
    val: typing.Any, element_type: type[T]
) -> typing.TypeGuard[Sequence[T]]:
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


def accepts_kwarg(func: Callable, name: str, *, strict: bool = True) -> bool:
    """Return True if `func` can be called with keyword argument `name`.

    Parameters
    ----------
    func
        The callable to inspect.
    name
        The name of the keyword argument to check for.
    strict
        If True, only return True if `name` is explicitly defined as a keyword parameter
        in `func`'s signature. If False, also return True if `func` accepts arbitrary
        keyword arguments via `**kwargs`.

    """
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        # Some builtins/C-extensions don't expose a signature reliably
        return False

    params = sig.parameters

    # Explicit keyword parameter?
    if name in params:
        p = params[name]
        return p.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    if strict:
        return False

    # Has **kwargs?
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
