"""Core logic for managing user settings.

This module implements the class `OptionManager` for managing user settings related to
the ImageTool. An instance of this class is created at the module level and can be
accessed as `options`. The settings are stored in a QSettings object, allowing for
persistent storage across application runs.
"""

import threading
import typing

from qtpy import QtCore

from erlab.interactive._options.schema import AppOptions


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
    return AppOptions.model_validate(
        _qsettings_to_dict(qsettings, AppOptions().model_dump())
    )


def write_settings(opts: AppOptions, qsettings: QtCore.QSettings) -> None:
    """Write settings from a pydantic model to QSettings.

    Parameters
    ----------
    opts : AppOptions
        An instance of AppOptions containing the settings to write.
    qsettings : QtCore.QSettings
        The QSettings object to write to.
    """
    _dict_to_qsettings(opts.model_dump(), qsettings)


class OptionManager:
    """Manager for application settings using QSettings."""

    def __init__(self) -> None:
        self._lock = threading.RLock()

    @property
    def qsettings(self) -> QtCore.QSettings:
        """Get the QSettings object."""
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
            qsettings.setValue(name, value)
            qsettings.sync()

    def __getitem__(self, name: str) -> typing.Any:
        return self.get(name)

    def __setitem__(self, name: str, value: typing.Any) -> None:
        self.set(name, value)


options = OptionManager()
