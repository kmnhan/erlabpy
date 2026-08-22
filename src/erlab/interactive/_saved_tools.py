"""Application registry for classes referenced by saved ToolWindow documents."""

from __future__ import annotations

import functools
import importlib.metadata
import typing

if typing.TYPE_CHECKING:
    from collections.abc import Callable

FIGURE_COMPOSER_TOOL_ID = "erlab.interactive._figurecomposer._tool:FigureComposerTool"
_ENTRY_POINT_GROUP = "erlab.interactive.tool_windows"

# A saved file can select only one of these application modules. Third-party tool
# classes remain supported after their package imports and registers the class through
# ToolWindow.__init_subclass__. Never derive an import target from document text.
_BUILTIN_TOOL_MODULES = {
    "erlab.interactive._fit1d:Fit1DTool": "erlab.interactive._fit1d",
    "erlab.interactive._fit2d:Fit2DTool": "erlab.interactive._fit2d",
    "erlab.interactive._mesh:MeshTool": "erlab.interactive._mesh",
    FIGURE_COMPOSER_TOOL_ID: "erlab.interactive._figurecomposer._tool",
    "erlab.interactive.derivative:DerivativeTool": "erlab.interactive.derivative",
    "erlab.interactive.fermiedge:GoldTool": "erlab.interactive.fermiedge",
    "erlab.interactive.fermiedge:ResolutionTool": "erlab.interactive.fermiedge",
    "erlab.interactive.kspace:KspaceTool": "erlab.interactive.kspace",
    "erlab.interactive.kspace:KspaceToolGUI": "erlab.interactive.kspace",
}
_SAVED_TOOL_CLASSES: dict[str, type] = {}
_SAVED_TOOL_LOADERS: dict[str, Callable[[], object]] = {}
_ENTRY_POINTS_DISCOVERED = False


def _load_builtin_tool(module_name: str, qualname: str) -> object:
    module = importlib.import_module(module_name)
    value: object = module
    for attribute in qualname.split("."):
        value = getattr(value, attribute)
    return value


def _register_saved_tool_loader(
    identifier: str,
    loader: Callable[[], object],
) -> None:
    """Register a loader supplied by the installed application."""
    _SAVED_TOOL_LOADERS[identifier] = loader
    _SAVED_TOOL_CLASSES.pop(identifier, None)


for _identifier, _module_name in _BUILTIN_TOOL_MODULES.items():
    _qualname = _identifier.partition(":")[2]
    _register_saved_tool_loader(
        _identifier,
        functools.partial(_load_builtin_tool, _module_name, _qualname),
    )


def _discover_saved_tool_entry_points() -> None:
    """Register locally installed tool extensions without using document text."""
    global _ENTRY_POINTS_DISCOVERED
    if _ENTRY_POINTS_DISCOVERED:
        return
    _ENTRY_POINTS_DISCOVERED = True
    for entry_point in importlib.metadata.entry_points(group=_ENTRY_POINT_GROUP):
        if entry_point.attr is None:
            continue
        identifier = f"{entry_point.module}:{entry_point.attr}"
        if identifier in _SAVED_TOOL_CLASSES or identifier in _SAVED_TOOL_LOADERS:
            continue
        _register_saved_tool_loader(
            identifier,
            entry_point.load,
        )


def register_saved_tool_class(cls: type) -> None:
    """Register the canonical identifier of an imported ToolWindow subclass."""
    identifier = f"{cls.__module__}:{cls.__qualname__}"
    _SAVED_TOOL_CLASSES[identifier] = cls
    _SAVED_TOOL_LOADERS.pop(identifier, None)


def resolve_saved_tool_class(identifier: str) -> type:
    """Resolve a class selected by the local registry or extension metadata."""
    _discover_saved_tool_entry_points()
    cls = _SAVED_TOOL_CLASSES.get(identifier)
    if cls is None and identifier in _SAVED_TOOL_LOADERS:
        value = _SAVED_TOOL_LOADERS[identifier]()
        if not isinstance(value, type):
            raise TypeError(f"Saved tool reference {identifier!r} is not a class")
        loaded_identifier = f"{value.__module__}:{value.__qualname__}"
        if loaded_identifier != identifier:
            raise TypeError(
                f"Saved tool reference {identifier!r} loaded {loaded_identifier!r}"
            )
        cls = value
        _SAVED_TOOL_CLASSES[identifier] = cls
        _SAVED_TOOL_LOADERS.pop(identifier, None)
    if cls is None:
        raise LookupError(f"Saved tool class {identifier!r} is not registered")
    return cls
