"""Core logic for managing user settings.

This module implements the class `OptionManager` for managing user settings related to
the ImageTool. An instance of this class is created at the module level and can be
accessed as `options`. The settings are stored in a QSettings object, allowing for
persistent storage across application runs.
"""

import os
import threading
import typing

import pydantic
from qtpy import QtCore

from erlab.interactive._options.schema import (
    AppOptions,
    normalize_workspace_compression_mode,
)

_SETTINGS_PATH_ENV_VAR = "ERLAB_INTERACTIVE_OPTIONS_PATH"
_WORKSPACE_COMPRESSION_PATH = "io/workspace/compression"
_LEGACY_WORKSPACE_COMPRESS_PATH = "io/workspace/compress"


def _qsettings_to_dict(
    qsettings: QtCore.QSettings, defaults: dict, prefix: str = ""
) -> dict[str, dict[str, typing.Any]]:
    """Read QSettings into a dictionary.

    This function reads settings recursively from a QSettings object, using a provided
    defaults dictionary to fill in any missing values.

    Parameters
    ----------
    qsettings : QtCore.QSettings
        The QSettings object to read from.
    defaults : dict
        A dictionary containing default values for the settings. The keys should match
        the keys in the QSettings object.
    prefix : str, optional
        A prefix to prepend to the keys when reading from QSettings. This is useful for
        namespacing settings. By default, it is an empty string.

    Returns
    -------
    dict
        A dictionary with the same structure as `defaults`, but with values read from
        QSettings where available.
    """
    result: dict[str, typing.Any] = {}
    for k, v in defaults.items():
        key = f"{prefix}/{k}" if prefix else k
        if isinstance(v, dict):
            result[k] = _qsettings_to_dict(qsettings, v, key)
        else:
            result[k] = qsettings.value(key, v)
    return result


def _dict_to_qsettings(d: dict, qsettings: QtCore.QSettings, prefix: str = "") -> None:
    """Write QSettings from a dictionary.

    This function writes settings recursively to a QSettings object, using the provided
    dictionary. If a key in the dictionary is a nested dictionary, it will be written
    with the prefix applied.

    Parameters
    ----------
    d : dict
        A dictionary containing the settings to write. The keys should match the keys in
        the QSettings object.
    qsettings : QtCore.QSettings
        The QSettings object to write to.
    prefix : str, optional
        A prefix to prepend to the keys when writing to QSettings. This is useful for
        namespacing settings. By default, it is an empty string.

    """
    for k, v in d.items():
        key = f"{prefix}/{k}" if prefix else k
        if isinstance(v, dict):
            _dict_to_qsettings(v, qsettings, key)
        elif v is None:
            qsettings.remove(key)
        else:
            qsettings.setValue(key, v)


def read_settings(qsettings: QtCore.QSettings) -> AppOptions:
    """Read settings from QSettings into a pydantic model.

    Parameters
    ----------
    qsettings : QtCore.QSettings
        The QSettings object to read from.

    Returns
    -------
    AppOptions
        An instance of AppOptions populated with values from QSettings.

    """
    values = _qsettings_to_dict(qsettings, AppOptions().model_dump())
    if not qsettings.contains(_WORKSPACE_COMPRESSION_PATH) and qsettings.contains(
        _LEGACY_WORKSPACE_COMPRESS_PATH
    ):
        values["io"]["workspace"]["compression"] = normalize_workspace_compression_mode(
            qsettings.value(_LEGACY_WORKSPACE_COMPRESS_PATH)
        )
    return AppOptions.model_validate(values)


def write_settings(opts: AppOptions, qsettings: QtCore.QSettings) -> None:
    """Write settings from a pydantic model to QSettings.

    Parameters
    ----------
    opts : AppOptions
        An instance of AppOptions containing the settings to write.
    qsettings : QtCore.QSettings
        The QSettings object to write to.
    """
    qsettings.remove(_LEGACY_WORKSPACE_COMPRESS_PATH)
    _dict_to_qsettings(opts.model_dump(), qsettings)


def option_value(model: AppOptions, name: str) -> typing.Any:
    """Return an option value by slash-separated path."""
    if name == _LEGACY_WORKSPACE_COMPRESS_PATH:
        return option_value(model, _WORKSPACE_COMPRESSION_PATH) != "none"
    value: typing.Any = model
    for key in name.split("/"):
        value = getattr(value, key)
    return value


def option_model_with_value(
    model: AppOptions, name: str, value: typing.Any
) -> AppOptions:
    """Return a copy of *model* with one slash-separated option path changed."""
    if name == _LEGACY_WORKSPACE_COMPRESS_PATH:
        name = _WORKSPACE_COMPRESSION_PATH
        value = normalize_workspace_compression_mode(value)
    data = model.model_dump()
    target = data
    keys = name.split("/")
    for key in keys[:-1]:
        child = target[key]
        if not isinstance(child, dict):
            raise KeyError(name)
        target = child
    target[keys[-1]] = value
    return AppOptions.model_validate(data)


def option_paths(model_cls: type[pydantic.BaseModel] = AppOptions) -> tuple[str, ...]:
    """Return all leaf option paths in schema order."""
    paths: list[str] = []

    def collect(cls: type[pydantic.BaseModel], prefix: str = "") -> None:
        default_instance = cls()
        for field_name in cls.model_fields:
            path = f"{prefix}/{field_name}" if prefix else field_name
            value = getattr(default_instance, field_name)
            if isinstance(value, pydantic.BaseModel):
                collect(type(value), path)
            else:
                paths.append(path)

    collect(model_cls)
    return tuple(paths)


def workspace_overridable_option_paths() -> tuple[str, ...]:
    """Return option paths allowed in ImageTool Manager workspace overrides."""
    paths: list[str] = []

    def collect(cls: type[pydantic.BaseModel], prefix: str = "") -> None:
        default_instance = cls()
        for field_name, field_info in cls.model_fields.items():
            path = f"{prefix}/{field_name}" if prefix else field_name
            value = getattr(default_instance, field_name)
            if isinstance(value, pydantic.BaseModel):
                collect(type(value), path)
                continue
            extra = getattr(field_info, "json_schema_extra", None) or {}
            if isinstance(extra, dict) and extra.get("workspace_overridable"):
                paths.append(path)

    collect(AppOptions)
    return tuple(paths)


def normalize_workspace_option_overrides(
    overrides: typing.Mapping[str, typing.Any] | None,
) -> dict[str, typing.Any]:
    """Keep only workspace-overridable option paths.

    Values are intentionally not fully validated here. A workspace may contain a loader
    or stylesheet that is unavailable on this computer; such values must remain in the
    saved override payload so they can become active again when available.
    """
    if not isinstance(overrides, typing.Mapping):
        return {}
    allowed = set(workspace_overridable_option_paths())
    return {
        str(path): value for path, value in overrides.items() if str(path) in allowed
    }


def model_with_workspace_overrides(
    user_options: AppOptions,
    overrides: typing.Mapping[str, typing.Any] | None,
) -> AppOptions:
    """Merge valid workspace overrides over user settings."""
    model = user_options
    for path, value in normalize_workspace_option_overrides(overrides).items():
        try:
            model = option_model_with_value(model, path, value)
        except (KeyError, TypeError, ValueError):
            continue
    return model


class OptionManager:
    """Manager for application settings using QSettings."""

    def __init__(self) -> None:
        self._lock = threading.RLock()

    @property
    def qsettings(self) -> QtCore.QSettings:
        """Get the QSettings object."""
        settings_path = os.environ.get(_SETTINGS_PATH_ENV_VAR)
        if settings_path:
            return QtCore.QSettings(
                settings_path,
                QtCore.QSettings.Format.IniFormat,
            )
        return QtCore.QSettings(
            QtCore.QSettings.Format.IniFormat,
            QtCore.QSettings.Scope.UserScope,
            "erlabpy",
            "interactive",
        )

    @property
    def model(self) -> AppOptions:
        with self._lock:
            return read_settings(self.qsettings)

    @model.setter
    def model(self, opt: AppOptions) -> None:
        with self._lock:
            qsettings = self.qsettings
            write_settings(opt, qsettings)
            qsettings.sync()

    def restore(self) -> None:
        """Restore settings to default values."""
        with self._lock:
            qsettings = self.qsettings
            qsettings.clear()
            write_settings(AppOptions(), qsettings)
            qsettings.sync()

    def get(self, name: str) -> typing.Any:
        """Get settings by name."""
        if name == _LEGACY_WORKSPACE_COMPRESS_PATH:
            return self.get(_WORKSPACE_COMPRESSION_PATH) != "none"
        keys = name.split("/")
        option: typing.Any = self.model.model_dump()
        for key in keys:
            if isinstance(option, dict) and key in option:
                option = option[key]
            else:
                option = None
                break
        return option

    def set(self, name: str, value: typing.Any) -> None:
        """Set settings by name and value."""
        with self._lock:
            qsettings = self.qsettings
            if name == _LEGACY_WORKSPACE_COMPRESS_PATH:
                qsettings.remove(_LEGACY_WORKSPACE_COMPRESS_PATH)
                if value is None:
                    qsettings.remove(_WORKSPACE_COMPRESSION_PATH)
                else:
                    qsettings.setValue(
                        _WORKSPACE_COMPRESSION_PATH,
                        normalize_workspace_compression_mode(value),
                    )
                qsettings.sync()
                return
            if value is None:
                qsettings.remove(name)
            else:
                qsettings.setValue(name, value)
            if name == _WORKSPACE_COMPRESSION_PATH:
                qsettings.remove(_LEGACY_WORKSPACE_COMPRESS_PATH)
            qsettings.sync()

    def __getitem__(self, name: str) -> typing.Any:
        return self.get(name)

    def __setitem__(self, name: str, value: typing.Any) -> None:
        self.set(name, value)


options = OptionManager()
