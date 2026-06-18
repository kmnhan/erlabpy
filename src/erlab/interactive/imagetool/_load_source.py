"""Private helpers for ImageTool file-load metadata and provenance."""

from __future__ import annotations

import importlib
import pathlib
import typing
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

import erlab
from erlab.interactive.imagetool import provenance

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

_FileDataSelection = provenance.FileDataSelection
_LoadFunc: typing.TypeAlias = tuple[
    Callable[..., typing.Any] | str,
    dict[str, typing.Any],
    int | dict[str, typing.Any] | _FileDataSelection,
]
_LoadKind: typing.TypeAlias = typing.Literal["erlab_loader", "callable"]


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


@dataclass(frozen=True)
class _ResolvedLoadFunc:
    """Normalized file loader metadata shared by UI, code, and replay paths.

    This is intentionally an internal value object. It records the result of resolving
    ``load_func`` once so callers do not independently rediscover the loader kind,
    display text, import/setup code, selected parsed array, and dtype cast behavior.
    """

    kind: _LoadKind
    target: str
    loader_label: str
    loader_text: str
    loader_expr: str
    imports: tuple[str, ...]
    setup_lines: tuple[str, ...]
    loader_name: str | None
    kwargs: dict[str, typing.Any]
    selection: provenance.FileDataSelection
    cast_float64: bool

    @property
    def kwargs_text(self) -> str:
        """Return human-readable kwargs text for the file-source details dialog."""
        if not self.kwargs:
            return "(none)"
        return erlab.interactive.utils.format_kwargs(
            typing.cast("dict[typing.Hashable, typing.Any]", self.kwargs)
        )

    def replay_call(
        self,
    ) -> provenance.FileReplayCall:
        """Return the serialized loader call used by structured file replay."""
        return provenance.FileReplayCall(
            kind=self.kind,
            target=self.target,
            kwargs=self.kwargs,
            selection=self.selection,
            cast_float64=self.cast_float64,
        )

    def load_code(self, file_path: Path, *, assign: str) -> str:
        """Return user-facing Python that reloads the same selected file data."""
        imports = list(self.imports)
        call_args = self._call_args(file_path)
        call_expr = f"{self.loader_expr}({', '.join(call_args)})"
        call_expr = _file_data_selection_code(call_expr, self.selection)
        if self.cast_float64:
            call_expr = f'{call_expr}.astype("float64")'

        lines = list(dict.fromkeys(imports))
        if lines:
            lines.append("")
        lines.extend(self.setup_lines)
        lines.append(f"{assign} = {call_expr}")
        return "\n".join(lines)

    def _call_args(self, file_path: Path) -> list[str]:
        """Return loader call arguments, using compact scan-number syntax when safe."""
        kwargs_str = _format_call_kwargs(self.kwargs)
        scan_call_args = (
            _scan_number_load_call_args(file_path, self.loader_name, self.kwargs)
            if self.loader_name is not None
            else None
        )
        if scan_call_args is not None:
            return scan_call_args
        call_args = [repr(str(file_path))]
        if kwargs_str:
            call_args.append(kwargs_str)
        return call_args


def _format_call_kwargs(kwargs: dict[str, typing.Any]) -> str:
    return (
        erlab.interactive.utils.format_call_kwargs(
            typing.cast("dict[typing.Hashable, typing.Any]", kwargs)
        )
        if kwargs
        else ""
    )


def _needs_float64_cast(source_input_dtype: np.dtype[typing.Any] | str | None) -> bool:
    return source_input_dtype is not None and np.dtype(source_input_dtype) not in (
        np.dtype(np.float32),
        np.dtype(np.float64),
    )


def _file_data_selection_code(
    load_expr: str,
    selection: provenance.FileDataSelection,
) -> str:
    if selection.kind == "dataarray":
        return load_expr
    if selection.kind == "dataset_variable":
        variable_code = provenance._provenance_value_code(selection.value)
        return f"{load_expr}[{variable_code}]"
    if selection.kind == "datatree_path":
        return f"{load_expr}[{selection.value!r}]"

    return (
        "erlab.interactive.imagetool.viewer_state._parse_input("
        f"{load_expr})[{selection.value}]"
    )


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


def _import_for_callable(loader: Callable[..., typing.Any], target: str) -> str:
    module = getattr(loader, "__module__", None)
    if isinstance(module, str) and target.startswith(f"{module}."):
        return f"import {module}"
    return f"import {target.split('.', maxsplit=1)[0]}"


def _resolve_load_func(
    load_func: _LoadFunc | None,
    *,
    source_input_dtype: np.dtype[typing.Any] | str | None = None,
) -> _ResolvedLoadFunc | None:
    if load_func is None:
        return None

    loader, kwargs, selection = load_func
    selection = _FileDataSelection.model_validate(selection)
    cast_float64 = _needs_float64_cast(source_input_dtype)
    if isinstance(loader, str):
        return _ResolvedLoadFunc(
            kind="erlab_loader",
            target=loader,
            loader_label="Loader",
            loader_text=loader,
            loader_expr="erlab.io.load",
            imports=(),
            setup_lines=(f"erlab.io.set_loader({loader!r})",),
            loader_name=loader,
            kwargs=kwargs,
            selection=selection,
            cast_float64=cast_float64,
        )

    func_instance = getattr(loader, "__self__", None)
    if isinstance(func_instance, erlab.io.dataloader.LoaderBase):
        loader_name = func_instance.name
        return _ResolvedLoadFunc(
            kind="erlab_loader",
            target=loader_name,
            loader_label="Loader",
            loader_text=loader_name,
            loader_expr="erlab.io.load",
            imports=(),
            setup_lines=(f"erlab.io.set_loader({loader_name!r})",),
            loader_name=loader_name,
            kwargs=kwargs,
            selection=selection,
            cast_float64=cast_float64,
        )

    target = _loader_callable_text(loader)
    if target is None:
        return None
    return _ResolvedLoadFunc(
        kind="callable",
        target=target,
        loader_label="Load Function",
        loader_text=target,
        loader_expr=target,
        imports=(_import_for_callable(loader, target),),
        setup_lines=(),
        loader_name=None,
        kwargs=kwargs,
        selection=selection,
        cast_float64=cast_float64,
    )


def _load_code_from_file_details(
    file_path: Path,
    load_func: _LoadFunc | None,
    *,
    assign: str = "data",
    source_input_dtype: np.dtype[typing.Any] | str | None = None,
) -> str | None:
    """Generate replay code for loading one ImageTool data source from a file."""
    if not erlab.interactive.utils._is_kwarg_name(assign):
        raise ValueError("assign must be a valid Python identifier")
    resolved = _resolve_load_func(
        load_func,
        source_input_dtype=source_input_dtype,
    )
    if resolved is None:
        return None
    return resolved.load_code(file_path, assign=assign)


def _fallback_load_source_label_and_text(
    load_func: _LoadFunc | None,
) -> tuple[str, str]:
    if load_func is None:
        return "Loader", "(unavailable)"
    loader = load_func[0]
    if isinstance(loader, str):
        return "Loader", loader
    return "Load Function", repr(loader)


def _load_source_label_and_text(
    load_func: _LoadFunc | None,
) -> tuple[str, str]:
    """Return a user-facing loader label and value for metadata display."""
    resolved = _resolve_load_func(load_func)
    if resolved is not None:
        return resolved.loader_label, resolved.loader_text
    return _fallback_load_source_label_and_text(load_func)


def _load_source_details(
    file_path: Path,
    load_func: _LoadFunc | None,
    resolved: _ResolvedLoadFunc | None,
) -> _LoadSourceDetails:
    if resolved is not None:
        return _LoadSourceDetails(
            path=file_path,
            loader_label=resolved.loader_label,
            loader_text=resolved.loader_text,
            kwargs_text=resolved.kwargs_text,
            load_code=resolved.load_code(file_path, assign="data"),
        )

    loader_label, loader_text = _fallback_load_source_label_and_text(load_func)
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
        load_code=None,
    )


def _load_source_details_from_file(
    file_path: Path,
    load_func: _LoadFunc | None,
    *,
    source_input_dtype: np.dtype[typing.Any] | str | None = None,
) -> _LoadSourceDetails:
    """Build manager metadata details from ImageTool file-load state."""
    resolved = _resolve_load_func(
        load_func,
        source_input_dtype=source_input_dtype,
    )
    return _load_source_details(file_path, load_func, resolved)


def _load_source_details_from_provenance(
    load_source: provenance.FileLoadSource,
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
    load_func: _LoadFunc | None,
    *,
    source_input_dtype: np.dtype[typing.Any] | str | None = None,
    replay_stages: Sequence[provenance.ReplayStage] = (),
) -> provenance.ToolProvenanceSpec | None:
    """Build replay provenance whose seed reloads the current data from disk."""
    resolved = _resolve_load_func(
        load_func,
        source_input_dtype=source_input_dtype,
    )
    if resolved is None:
        return None
    details = _load_source_details(file_path, load_func, resolved)
    return provenance.file_load(
        start_label=f"Load data from file {file_path.name!r}",
        seed_code=resolved.load_code(file_path, assign="derived"),
        file_load_source=provenance.FileLoadSource(
            path=str(details.path),
            loader_label=details.loader_label,
            loader_text=details.loader_text,
            kwargs_text=details.kwargs_text,
            replay_call=resolved.replay_call(),
            load_code=details.load_code,
        ),
        replay_stages=replay_stages,
    )
