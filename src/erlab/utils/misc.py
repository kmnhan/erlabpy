import functools
import importlib
import inspect
import pathlib
import sys
import warnings
from types import ModuleType
from typing import Any, overload

import numpy as np

_NestedGeneric = np.generic | list["_NestedGeneric"] | Any


@overload
def _convert_to_native(obj: np.generic) -> Any: ...
@overload
def _convert_to_native(obj: list[_NestedGeneric]) -> list[Any]: ...
@overload
def _convert_to_native(obj: Any) -> Any: ...
def _convert_to_native(obj: _NestedGeneric) -> Any:
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


class LazyImport:
    """Lazily import a module when an attribute is accessed.

    Used to delay the import of a module until it is actually needed.

    Parameters
    ----------
    module_name : str
        The name of the module to be imported lazily.
    err_msg : str, optional
        If present, this message will be displayed in the ImportError raised when the
        accessed module is not found.

    Examples
    --------
    >>> np = LazyImport("numpy")
    >>> np.array([1, 2, 3])
    array([1, 2, 3])

    """

    def __init__(self, module_name: str, err_msg: str | None) -> None:
        self._module_name = module_name
        self._err_msg = err_msg

    def __getattr__(self, item: str) -> Any:
        return getattr(self._module, item)

    @functools.cached_property
    def _module(self) -> ModuleType:
        if (self._err_msg is not None) and (
            not importlib.util.find_spec(self._module_name)
        ):
            raise ImportError(self._err_msg)

        return importlib.import_module(self._module_name)
