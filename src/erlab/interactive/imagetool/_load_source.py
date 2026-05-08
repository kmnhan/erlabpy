"""Private helpers for ImageTool file-load metadata and provenance."""

from __future__ import annotations

import importlib
import pathlib
import typing
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

import erlab

if typing.TYPE_CHECKING:
    import os
    from pathlib import Path


_LOAD_KWARGS_NOT_FOR_IDENTIFY = frozenset(
    {
        "chunks",
        "single",
        "combine",
        "parallel",
        "progress",
        "load_kwargs",
        "loader_extensions",
    }
)

_RESERVED_REPLAY_SOURCE_NAMES = {
    "data",
    "derived",
    "result",
    "target",
    "model",
    "_fit_data",
    "_processed",
}


@dataclass(frozen=True)
class _LoadSourceDetails:
    """Display and copy-code details for data loaded from a file.

    The manager uses this model for metadata dialogs, while ImageTool uses the same
    load information to build replay provenance for file-backed data.
    """

    path: Path
    loader_label: str
    loader_text: str
    kwargs_text: str
    load_code: str | None


def _resolve_identified_path(
    path: str | os.PathLike[str], data_dir: pathlib.Path
) -> pathlib.Path:
    """Resolve a loader-identified path relative to its data directory."""
    resolved = pathlib.Path(path)
    if not resolved.is_absolute():
        resolved = data_dir / resolved
    return resolved.resolve()


def _scan_number_load_call_args(
    file_path: Path,
    loader_name: str,
    kwargs: dict[str, typing.Any],
) -> list[str] | None:
    """Return compact ``erlab.io.load(scan, data_dir=...)`` args when unambiguous.

    ERLab loaders can often recover the original scan number from a file name. This
    keeps generated provenance code closer to the code users would normally write,
    but only when identifying the inferred scan resolves back to ``file_path``.
    """
    if kwargs.get("single", False) or "data_dir" in kwargs:
        return None
    if any(not isinstance(key, str) for key in kwargs):
        return None
    if loader_name not in erlab.io.loaders:
        return None

    loader = erlab.io.loaders[loader_name]
    try:
        scan_num, infer_kwargs = loader.infer_index(file_path.stem)
    except Exception:
        return None

    if isinstance(scan_num, bool) or not isinstance(scan_num, int | np.integer):
        return None
    if infer_kwargs is None:
        infer_kwargs = {}
    if not isinstance(infer_kwargs, Mapping):
        return None

    scan_num = int(scan_num)
    infer_kwargs = dict(infer_kwargs)
    if "data_dir" in infer_kwargs or any(
        not isinstance(key, str) for key in infer_kwargs
    ):
        return None

    data_dir = pathlib.Path(file_path.parent)
    identify_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key not in _LOAD_KWARGS_NOT_FOR_IDENTIFY
    } | infer_kwargs
    try:
        identified = loader.identify(scan_num, data_dir, **identify_kwargs)
    except Exception:
        return None
    if identified is None:
        return None

    identified_paths = identified[0]
    target_path = pathlib.Path(file_path).resolve()
    if not any(
        _resolve_identified_path(path, data_dir) == target_path
        for path in identified_paths
    ):
        return None

    load_kwargs = kwargs | infer_kwargs
    call_args = [repr(scan_num), f"data_dir={str(data_dir)!r}"]
    if load_kwargs:
        call_args.append(
            erlab.interactive.utils.format_call_kwargs(
                typing.cast("dict[typing.Hashable, typing.Any]", load_kwargs)
            )
        )
    return call_args


def _load_code_from_file_details(
    file_path: Path,
    load_func: tuple[Callable[..., typing.Any] | str, dict[str, typing.Any], int]
    | None,
    *,
    assign: str = "data",
    source_input_dtype: np.dtype[typing.Any] | str | None = None,
) -> str | None:
    """Generate replay code for loading one ImageTool data source from a file."""
    if not erlab.interactive.utils._is_kwarg_name(assign):
        raise ValueError("assign must be a valid Python identifier")
    if load_func is None or load_func[2] != 0:
        return None

    imports: list[str] = []
    setup_lines: list[str] = []
    loader = load_func[0]
    loader_name = None
    if isinstance(loader, str):
        loader_name = loader
    else:
        func_instance = getattr(loader, "__self__", None)
        if isinstance(func_instance, erlab.io.dataloader.LoaderBase):
            loader_name = func_instance.name

    if loader_name is not None:
        imports.append("import erlab")
        setup_lines.append(f"erlab.io.set_loader({loader_name!r})")
        loader_expr = "erlab.io.load"
    else:
        if isinstance(loader, str):
            return None
        callable_loader_expr = _loader_callable_text(loader)
        if callable_loader_expr is None:
            return None
        loader_expr = callable_loader_expr
        loader_module = getattr(loader, "__module__", None)
        if isinstance(loader_module, str) and callable_loader_expr.startswith(
            f"{loader_module}."
        ):
            imports.insert(0, f"import {loader_module}")
        else:
            imports.insert(0, f"import {callable_loader_expr.split('.', 1)[0]}")

    kwargs = load_func[1]
    kwargs_str = (
        erlab.interactive.utils.format_call_kwargs(
            typing.cast("dict[typing.Hashable, typing.Any]", kwargs)
        )
        if kwargs
        else ""
    )
    call_args = (
        _scan_number_load_call_args(file_path, loader_name, kwargs)
        if loader_name is not None
        else None
    )
    if call_args is None:
        call_args = [repr(str(file_path))]
        if kwargs_str:
            call_args.append(kwargs_str)
    imports = list(dict.fromkeys(imports))
    call_expr = f"{loader_expr}({', '.join(call_args)})"
    if source_input_dtype is not None and np.dtype(source_input_dtype) not in (
        np.dtype(np.float32),
        np.dtype(np.float64),
    ):
        call_expr = f'{call_expr}.astype("float64")'

    return "\n".join(
        [
            *imports,
            "",
            *setup_lines,
            f"{assign} = {call_expr}",
        ]
    )


def _load_source_label_and_text(
    load_func: tuple[Callable[..., typing.Any] | str, dict[str, typing.Any], int]
    | None,
) -> tuple[str, str]:
    """Return a user-facing loader label and value for metadata display."""
    if load_func is None:
        return "Loader", "(unavailable)"

    loader = load_func[0]
    if isinstance(loader, str):
        return "Loader", loader

    loader_text = _loader_callable_text(loader)
    if loader_text is None:
        return "Load Function", repr(loader)
    return "Load Function", loader_text


def _loader_callable_text(loader: Callable[..., typing.Any]) -> str | None:
    """Return importable text for a loader callable when it has a stable path."""
    module = getattr(loader, "__module__", None)
    qualname = getattr(loader, "__qualname__", getattr(loader, "__name__", None))
    if module is None or qualname is None or "<locals>" in qualname:
        return None

    # Prefer a public top-level alias when the callable is re-exported there so the
    # manager shows stable, user-facing load code instead of private module paths.
    top_level = module.split(".", maxsplit=1)[0]
    try:
        package = importlib.import_module(top_level)
    except Exception:
        return f"{module}.{qualname}"

    target: typing.Any = package
    for attr in qualname.split("."):
        target = getattr(target, attr, None)
        if target is None:
            break
    if target is loader:
        return f"{top_level}.{qualname}"
    return f"{module}.{qualname}"


def _load_source_details_from_file(
    file_path: Path,
    load_func: tuple[Callable[..., typing.Any] | str, dict[str, typing.Any], int]
    | None,
    *,
    source_input_dtype: np.dtype[typing.Any] | str | None = None,
) -> _LoadSourceDetails:
    """Build manager metadata details from ImageTool file-load state."""
    loader_label, loader_text = _load_source_label_and_text(load_func)
    kwargs_text = "(unavailable)"
    if load_func is not None:
        kwargs_text = (
            erlab.interactive.utils.format_kwargs(
                typing.cast("dict[typing.Hashable, typing.Any]", load_func[1])
            )
            if load_func[1]
            else "(none)"
        )
    return _LoadSourceDetails(
        path=file_path,
        loader_label=loader_label,
        loader_text=loader_text,
        kwargs_text=kwargs_text,
        load_code=_load_code_from_file_details(
            file_path,
            load_func,
            source_input_dtype=source_input_dtype,
        ),
    )


def _load_source_details_from_provenance(
    load_source: erlab.interactive.imagetool.provenance.FileLoadSource,
) -> _LoadSourceDetails:
    """Build manager metadata details from serialized provenance file metadata."""
    return _LoadSourceDetails(
        path=pathlib.Path(load_source.path),
        loader_label=load_source.loader_label,
        loader_text=load_source.loader_text,
        kwargs_text=load_source.kwargs_text,
        load_code=load_source.load_code,
    )


def _default_load_source_name(file_path: Path) -> str:
    """Choose a non-conflicting variable name for copied file-load code."""
    name = erlab.interactive.utils.IdentifierValidator().fixup(file_path.stem)
    return "source_data" if name in _RESERVED_REPLAY_SOURCE_NAMES else name


def _load_provenance_from_file_details(
    file_path: Path,
    load_func: tuple[Callable[..., typing.Any] | str, dict[str, typing.Any], int]
    | None,
    *,
    source_input_dtype: np.dtype[typing.Any] | str | None = None,
) -> erlab.interactive.imagetool.provenance.ToolProvenanceSpec | None:
    """Build replay provenance whose seed reloads the current data from disk."""
    details = _load_source_details_from_file(
        file_path,
        load_func,
        source_input_dtype=source_input_dtype,
    )
    seed_code = _load_code_from_file_details(
        file_path,
        load_func,
        assign="derived",
        source_input_dtype=source_input_dtype,
    )
    if seed_code is None:
        return None
    return erlab.interactive.imagetool.provenance.script(
        start_label=f"Load data from file {file_path.name!r}",
        seed_code=seed_code,
        active_name="derived",
        file_load_source=erlab.interactive.imagetool.provenance.FileLoadSource(
            path=details.path,
            loader_label=details.loader_label,
            loader_text=details.loader_text,
            kwargs_text=details.kwargs_text,
            load_code=details.load_code,
        ),
    )
