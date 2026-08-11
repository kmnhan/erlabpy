"""Simple analysis and loader extensions for ERLab.

Extension authors normally need only :func:`routine` or :func:`loader`. Decorated
functions remain ordinary Python functions and work directly in notebooks.
"""

from erlab.extensions._api import (
    load_entry_point,
    load_script,
    loader,
    routine,
    run_loader,
    run_routine,
)
from erlab.extensions._models import (
    EXTENSION_API_VERSION,
    CapabilityDescriptor,
    ExtensionError,
    ExtensionExecutionError,
    ExtensionImportError,
    ExtensionNotFoundError,
    ExtensionSignatureError,
    LoadedEntryPoint,
    LoadedScript,
    LoaderDescriptor,
    ParameterDescriptor,
    ParameterKind,
    RoutineDescriptor,
)

__all__ = (
    "EXTENSION_API_VERSION",
    "CapabilityDescriptor",
    "ExtensionError",
    "ExtensionExecutionError",
    "ExtensionImportError",
    "ExtensionNotFoundError",
    "ExtensionSignatureError",
    "LoadedEntryPoint",
    "LoadedScript",
    "LoaderDescriptor",
    "ParameterDescriptor",
    "ParameterKind",
    "RoutineDescriptor",
    "load_entry_point",
    "load_script",
    "loader",
    "routine",
    "run_loader",
    "run_routine",
)
