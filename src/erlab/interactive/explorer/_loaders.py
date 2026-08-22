"""Loader adapters used by Data Explorer."""

from __future__ import annotations

import typing

import xarray as xr

from erlab.interactive._file_loaders import (
    BUILTIN_FILE_LOADER_SPECS,
    BuiltinFileLoaderSpec,
)

if typing.TYPE_CHECKING:
    import os


class _BuiltinExplorerLoader:
    """Adapt a direct callable loader to the Data Explorer loader contract."""

    always_single = True

    def __init__(self, spec: BuiltinFileLoaderSpec) -> None:
        self.spec = spec
        self.name = spec.id
        self.display_name = spec.label
        self.description = spec.description
        self.extensions = set(spec.extensions)

    def load(
        self,
        file_path: str | os.PathLike[str],
        *,
        single: bool = True,
        load_kwargs: dict[str, typing.Any] | None = None,
        **kwargs: typing.Any,
    ) -> xr.DataArray:
        """Load one DataArray for metadata display or graphical preview."""
        del single
        options = dict(self.spec.default_kwargs)
        options.update(load_kwargs or {})
        options.update(kwargs)
        without_values = bool(options.pop("without_values", False))
        load_func = xr.open_dataarray if without_values else self.spec.load_func
        return load_func(file_path, **options)

    def option_overrides(self, kwargs: dict[str, typing.Any]) -> dict[str, typing.Any]:
        """Return user options without unchanged built-in defaults."""
        return {
            key: value
            for key, value in kwargs.items()
            if key not in self.spec.default_kwargs
            or value != self.spec.default_kwargs[key]
        }


BUILTIN_EXPLORER_LOADERS = {
    spec.id: _BuiltinExplorerLoader(spec) for spec in BUILTIN_FILE_LOADER_SPECS
}
