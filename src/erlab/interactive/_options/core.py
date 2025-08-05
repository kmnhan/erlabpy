"""Core logic for managing user settings.

This module implements the class `OptionManager` for managing user settings related to
the ImageTool. An instance of this class is created at the module level and can be
accessed as `options`. The settings are stored in a QSettings object, allowing for
persistent storage across application runs.
"""

import typing

from qtpy import QtCore

from erlab.interactive._options.defaults import DEFAULT_OPTIONS, _as_bool, _as_float


def read_settings(
    qsettings: QtCore.QSettings, defaults: dict, prefix: str = ""
) -> dict[str, dict[str, typing.Any]]:
    """Parse QSettings into a dictionary with a structure matching `defaults`.

    This function reads settings recursively from a QSettings object, using the provided
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
            result[k] = read_settings(qsettings, v, key)
        else:
            # Convert to appropriate type based on the default value
            if isinstance(v, bool):
                result[k] = _as_bool(qsettings.value(key, v))
            elif isinstance(v, float):
                result[k] = _as_float(qsettings.value(key, v))
            else:
                result[k] = qsettings.value(key, v)
    return result


def write_settings(d: dict, qsettings: QtCore.QSettings, prefix: str = "") -> None:
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
            write_settings(v, qsettings, key)
        else:
            qsettings.setValue(key, v)


class OptionManager:
    """Manager for application settings using QSettings.

    This class provides an interface to read and write settings as a dictionary. The
    settings are stored in a QSettings object, which allows for persistent storage
    across application runs. The settings can be restored to defaults.
    """

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
    def option_dict(self) -> dict:
        """Get the current settings as a dictionary."""
        return read_settings(self.qsettings, DEFAULT_OPTIONS)

    @option_dict.setter
    def option_dict(self, d: dict) -> None:
        """Set the settings from a dictionary."""
        write_settings(d, self.qsettings)
        self.qsettings.sync()

    def restore(self) -> None:
        """Restore the settings to defaults."""
        self.qsettings.clear()
        self.option_dict = DEFAULT_OPTIONS

    def get(self, name: str) -> typing.Any:
        """Get settings by name."""
        keys = name.split("/")
        option: typing.Any = self.option_dict
        for key in keys:
            if isinstance(option, dict) and key in option:
                option = option[key]
            else:
                option = None
                break
        return option

    def set(self, name: str, value: typing.Any) -> None:
        """Set settings by name."""
        self.qsettings.setValue(name, value)

    def __getitem__(self, name: str) -> typing.Any:
        return self.get(name)

    def __setitem__(self, name: str, value: typing.Any) -> None:
        """Set settings by name."""
        self.set(name, value)


options = OptionManager()
