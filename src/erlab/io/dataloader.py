r"""Base functionality for implementing data loaders.

This module provides a base class :class:`LoaderBase` for implementing data loaders.
Data loaders are plugins used to load data from various file formats.

Each data loader is a subclass of :class:`LoaderBase` that must implement several
methods and attributes.

A detailed guide on how to implement a data loader can be found in the :ref:`User Guide
<implementing-plugins>`.
"""

from __future__ import annotations

__all__ = [
    "LoaderBase",
    "LoaderNotFoundError",
    "LoaderRegistry",
    "UnsupportedFileError",
    "ValidationError",
    "ValidationWarning",
]

import contextlib
import errno
import importlib
import itertools
import os
import pathlib
import typing
import warnings

import numpy as np
import numpy.typing as npt
import pandas
import xarray as xr

import erlab

if typing.TYPE_CHECKING:
    import datetime
    from collections.abc import (
        Callable,
        ItemsView,
        Iterable,
        Iterator,
        KeysView,
        Mapping,
        Sequence,
    )

    import joblib
    import tqdm.auto as tqdm
else:
    import lazy_loader as _lazy

    from erlab.utils.misc import LazyImport

    joblib = _lazy.load("joblib")
    tqdm = LazyImport("tqdm.auto")


class ValidationWarning(UserWarning):
    """Issued when the loaded data fails validation checks."""


class ValidationError(Exception):
    """Raised when the loaded data fails validation checks."""


class LoaderNotFoundError(Exception):
    """Raised when a loader is not found in the registry."""

    def __init__(self, key: str) -> None:
        super().__init__(f"Loader for name or alias {key} not found in the registry")


class UnsupportedFileError(Exception):
    """Raised when the loader does not support the given file extension."""

    def __init__(self, loader: LoaderBase, file_path: pathlib.Path) -> None:
        extensions = loader.extensions
        if extensions is not None:
            super().__init__(self._make_msg(loader.name, file_path, extensions))

    @staticmethod
    def _make_msg(
        loader_name: str, file_path: pathlib.Path, extensions: set[str]
    ) -> str:
        ext = file_path.suffix.lower()
        ext_sorted = sorted(f"'{ext}'" for ext in extensions)

        file_desc: str = "Directories" if file_path.is_dir() else f"'{ext}' files"

        return (
            f"{file_desc} are not supported by loader '{loader_name}'."
            f" Supported file types are: {', '.join(ext_sorted)}"
        )


class _Loader(type):
    """Metaclass for data loaders.

    This metaclass wraps the `identify` method to display informative warnings and error
    messages for missing files or multiple files found for a single scan.
    """

    def __new__(cls, name, bases, dct):
        new_class = super().__new__(cls, name, bases, dct)

        # Wrap identify
        if "identify" in dct:
            original_identify = dct["identify"]

            def wrapped_identify(self, num, data_dir, **kwargs):
                result = original_identify(self, num, data_dir, **kwargs)
                if result is None or (result is not None and len(result[0]) == 0):
                    dirname = data_dir or "the working directory"
                    msg = (
                        f"{self.__class__.__name__}.identify found no files "
                        f"for scan {num} in {dirname}"
                    )
                    if len(kwargs) > 0:
                        msg += f" with additional arguments {kwargs}"
                    raise FileNotFoundError(msg)

                if self.always_single and len(result[0]) > 1:
                    erlab.utils.misc.emit_user_level_warning(
                        f"Multiple files found for scan {num}, using {result[0][0]}"
                    )
                    return result[0][:1], result[1]
                return result

            wrapped_identify.__doc__ = original_identify.__doc__
            new_class.identify = wrapped_identify

            # For linkcode to work in the documentation
            # See linkcode_resolve in docs/conf.py
            new_class._original_identify = original_identify

        if "load_single" in dct:
            original_load_single = dct["load_single"]

            def wrapped_load_single(self, file_path, **kwargs):
                file_path = pathlib.Path(file_path)

                if not file_path.exists():
                    raise FileNotFoundError(
                        errno.ENOENT, os.strerror(errno.ENOENT), file_path
                    )

                if file_path.suffix.lower() not in self.extensions:
                    raise UnsupportedFileError(self, file_path)
                return original_load_single(self, file_path, **kwargs)

            wrapped_load_single.__doc__ = original_load_single.__doc__
            new_class.load_single = wrapped_load_single

            # For linkcode to work in the documentation
            # See linkcode_resolve in docs/conf.py
            new_class._original_load_single = original_load_single

        return new_class


class LoaderBase(metaclass=_Loader):
    """Base class for loader plugins."""

    name: str
    """
    Name of the loader. Using a unique and descriptive name is recommended. For easy
    access, it is recommended to use a name that passes :meth:`str.isidentifier`.

    Notes
    -----
    - Changing the name of a loader is not recommended as it may break existing code.
      Pick a simple, descriptive name that is unlikely to change.
    - Loaders with the name prefixed with an underscore are not registered.
    """

    description: str
    """A short description of the loader shown to users.

    .. versionadded:: 3.3.0
    """

    aliases: Iterable[str] | None = None
    """Alternative names for the loader.

    .. deprecated:: 3.3.0

       Accessing loaders with aliases is deprecated and will be removed in a future
       version. Use the loader name instead.

    """

    extensions: typing.ClassVar[set[str] | None] = None
    """File extensions supported by the loader in lowercase with the leading dot.

    An `UnsupportedFileError` is raised if a file with an unsupported extension is
    passed to the loader. If `None`, the loader will attempt to load any file passed to
    it.

    If the loader supports directories, the extension should be an empty string.

    .. versionadded:: 3.5.1
    """

    name_map: typing.ClassVar[dict[str, str | Iterable[str]]] = {}
    """
    Dictionary that maps **new** coordinate or attribute names to **original**
    coordinate or attribute names. If there are multiple possible names for a single
    attribute, the value can be passed as an iterable.

    Note
    ----
    - Non-dimension coordinates in the resulting data will try to follow the order of
      the keys in this mapping.
    - Original **coordinate** names included in this mapping will be replaced by the new
      names. However, original **attribute** names will be duplicated with the new names
      so that both the original and new names are present in the data after loading.
      This is to keep track of the original names for reference.
    """

    coordinate_attrs: tuple[str, ...] = ()
    """
    Attribute names (after renaming) that should be treated as coordinates.

    Put any attributes that should be propagated when concatenating data here.

    Notes
    -----
    - If a listed attribute is not found, it is silently skipped.
    - The attributes given here, both before and after renaming, are removed from the
      attributes to avoid conflicting values.
    - If an existing coordinate with the same name is already present, the existing
      coordinate takes precedence and the attribute is silently dropped.

    See Also
    --------
    :meth:`process_keys <erlab.io.dataloader.LoaderBase.process_keys>`
    """

    average_attrs: tuple[str, ...] = ()
    """
    Names of attributes or coordinates (after renaming) that should be averaged over.

    This is useful for attributes that may slightly vary between scans.

    Notes
    -----
    - If a listed attribute is not found, it is silently skipped.
    - Attributes listed here are first treated as coordinates in :meth:`process_keys
      <erlab.io.dataloader.LoaderBase.process_keys>`, and then averaged in
      :meth:`post_process <erlab.io.dataloader.LoaderBase.post_process>`.

    See Also
    --------
    :meth:`process_keys <erlab.io.dataloader.LoaderBase.process_keys>`,
    :meth:`post_process <erlab.io.dataloader.LoaderBase.post_process>`
    """

    additional_attrs: typing.ClassVar[
        dict[
            str,
            str
            | float
            | datetime.datetime
            | Callable[[xr.DataArray], str | float | datetime.datetime],
        ]
    ] = {}
    """Additional attributes to be added to the data after loading.

    If a callable is provided, it will be called with the data as the only argument.

    Notes
    -----
    - The attributes are added after renaming with :meth:`process_keys
      <erlab.io.dataloader.LoaderBase.process_keys>`, so keys will appear in the data as
      provided.
    - If an attribute with the same name is already present in the data, it is skipped
      unless the key is listed in :attr:`overridden_attrs
      <erlab.io.dataloader.LoaderBase.overridden_attrs>`.
    """

    overridden_attrs: tuple[str, ...] = ()
    """Keys in :attr:`additional_attrs` that should override existing attributes."""

    additional_coords: typing.ClassVar[
        dict[
            str,
            str
            | float
            | datetime.datetime
            | Callable[[xr.DataArray], str | float | datetime.datetime],
        ]
    ] = {}
    """Additional coordinates to be added to the data after loading.

    If a callable is provided, it will be called with the data as the only argument.

    Notes
    -----
    - The coordinates are added after renaming with :meth:`process_keys
      <erlab.io.dataloader.LoaderBase.process_keys>`, so keys will appear in the data as
      provided.
    - If a coordinate with the same name is already present in the data, it is skipped
      unless the key is listed in :attr:`overridden_coords
      <erlab.io.dataloader.LoaderBase.overridden_coords>`.
    """

    overridden_coords: tuple[str, ...] = ()
    """Keys in :attr:`additional_coords` that should override existing coordinates."""

    always_single: bool = True
    """
    Setting this to `True` disables implicit loading of multiple files for a single
    scan. This is useful for setups where each scan is always stored in a single file.
    """

    parallel_threshold: int = 30
    """
    Minimum number of files in a scan to use parallel loading. If the number of files is
    less than this threshold, files are loaded sequentially.

    Only used when :attr:`always_single <erlab.io.dataloader.LoaderBase.always_single>`
    is `False`.
    """

    skip_validate: bool = False
    """
    If `True`, validation checks will be skipped. If `False`, data will be checked with
    :meth:`validate <erlab.io.dataloader.LoaderBase.validate>`.
    """

    strict_validation: bool = False
    """
    If `True`, validation checks will raise a `ValidationError` on the first failure
    instead of warning. Useful for debugging data loaders. This has no effect if
    `skip_validate` is `True`.
    """

    formatters: typing.ClassVar[dict[str, Callable]] = {}
    """Optional mapping from attr or coord names (after renaming) to custom formatters.

    The formatters are callables that takes the attribute value and returns a value that
    can be converted to a string via :meth:`value_to_string
    <erlab.io.dataloader.LoaderBase.value_to_string>`. The resulting string
    representations are used for human readable display in the summary table and the
    information accessor.

    The values returned by the formatters will be further formatted by
    :meth:`value_to_string <erlab.io.dataloader.LoaderBase.value_to_string>` before
    being displayed.

    If the key is a coordinate, the function will automatically be vectorized over every
    value.

    Note
    ----
    The formatters are only used for display purposes and do not affect the stored data.

    See Also
    --------
    :meth:`get_formatted_attr_or_coord`
        The method that uses this mapping to provide human-readable values.
    """

    summary_sort: str | None = None
    """Optional default column to sort the summary table by.

    If `None`, the summary table is sorted in the order of the files returned by
    :meth:`files_for_summary <erlab.io.dataloader.LoaderBase.files_for_summary>`.
    """

    @property
    def summary_attrs(self) -> dict[str, str | Callable[[xr.DataArray], typing.Any]]:
        """Mapping from summary column names to attr or coord names (after renaming).

        If the value is a callable, it will be called with the data as the only
        argument. This can be used to extract values from the data that are not stored
        as attributes or spread across multiple attributes.

        If not overridden, returns a basic mapping based on :attr:`name_map`.

        It is highly recommended to override this property to provide a more detailed
        and informative summary. See existing loaders for examples.

        """
        excluded = {"eV", "alpha", "sample_workfunction"}
        return {k: k for k in self.name_map if k not in excluded}

    @property
    def _name_map_reversed(self) -> dict[str, str]:
        """Reverse of :attr:`name_map <erlab.io.dataloader.LoaderBase.name_map>`.

        Returns a mapping from original names to new names.
        """
        return self._reverse_mapping(self.name_map)

    @property
    def file_dialog_methods(self) -> dict[str, tuple[Callable, dict[str, typing.Any]]]:
        """Map from file dialog names to the loader method and its arguments.

        Override this property in the subclass to provide support for loading data from
        the load menu of the ImageTool GUI.

        Returns
        -------
        loader_mapping : dictionary of str to tuple of (callable, dict)
            A dictionary mapping the file dialog names to a tuple of length 2 containing
            the data loading function and arguments.

            The keys should be the names of the file dialog options passed to
            :meth:`QtWidgets.QFileDialog.setNameFilter`.

            The first item of the value tuple should be a callable that takes the first
            positional argument as a path to a file, usually ``self.load``.

            The second item should be a dictionary containing keyword arguments to be
            passed to the method.

            Multiple key-value pairs can be returned to provide multiple options.

        Example
        -------
        For instance, the loader for ALS BL4 implements the following mapping which
        enables loading ``.pxt`` and ``.ibw`` files within ImageTool using ``self.load``
        with no keyword arguments::

                @property
                def file_dialog_methods(self):
                    return {"ALS BL4.0.3 Raw Data (*.pxt, *.ibw)": (self.load, {})}

        """
        return {}

    @staticmethod
    def _reverse_mapping(mapping: Mapping[str, str | Iterable[str]]) -> dict[str, str]:
        """Reverse the given mapping dictionary to form a one-to-one mapping.

        Parameters
        ----------
        mapping
            The mapping dictionary to be reversed.

        Example
        -------

        >>> mapping = {"a": "1", "b": ["2", "3"]}
        >>> reverse_mapping(mapping)
        {'1': 'a', '2': 'b', '3': 'b'}

        """
        out: dict[str, str] = {}
        for k, v in mapping.items():
            if isinstance(v, str):
                out[v] = k
            else:
                for vi in v:
                    out[vi] = k
        return out

    @classmethod
    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)

        if not hasattr(cls, "name"):
            raise NotImplementedError("name attribute must be defined in the subclass")

        if not cls.name.startswith("_"):
            LoaderRegistry.instance()._register(cls)

    @classmethod
    def value_to_string(cls, val: object) -> str:
        """Format the given value based on its type.

        The default behavior formats the given value with
        :func:`erlab.utils.formatting.format_value`. Override this classmethod to change
        the printed format of summaries and information accessors. This method is
        applied after the formatters in :attr:`formatters
        <erlab.io.dataloader.LoaderBase.formatters>`.

        """
        return erlab.utils.formatting.format_value(val)

    @classmethod
    def get_styler(cls, df: pandas.DataFrame) -> pandas.io.formats.style.Styler:
        """Return a styled version of the given dataframe.

        This method, along with :meth:`value_to_string
        <erlab.io.dataloader.LoaderBase.value_to_string>`, determines the display
        formatting of the summary dataframe. Override this classmethod to change the
        display style.

        Parameters
        ----------
        df
            The summary dataframe.

        Returns
        -------
        pandas.io.formats.style.Styler
            The styler to be displayed.

        """
        style = df.style.format(cls.value_to_string)

        hidden = [c for c in ("Path",) if c in df.columns]
        if len(hidden) > 0:
            style = style.hide(hidden, axis="columns")

        return style

    def load(
        self,
        identifier: str | os.PathLike | int,
        data_dir: str | os.PathLike | None = None,
        *,
        single: bool = False,
        combine: bool = True,
        parallel: bool | None = None,
        progress: bool = True,
        load_kwargs: dict[str, typing.Any] | None = None,
        **kwargs,
    ) -> (
        xr.DataArray
        | xr.Dataset
        | xr.DataTree
        | list[xr.DataArray]
        | list[xr.Dataset]
        | list[xr.DataTree]
    ):
        """Load ARPES data.

        This method is the main entry point for loading ARPES data.

        .. note::

            This method is not meant to be overriden in subclasses.

        Parameters
        ----------
        identifier
            Value that identifies a scan uniquely.

            - If a string or path-like object is given, it is assumed to be the path to
              the data file relative to `data_dir`. If `data_dir` is not specified,
              `identifier` is assumed to be the full path to the data file.

            - If an integer is given, it is assumed to be a number that specifies the
              scan number, and is used to automatically determine the path to the data
              file(s). In this case, the `data_dir` argument must be specified.
        data_dir
            Where to look for the data. Must be a path to a valid directory. This
            argument is required when `identifier` is an integer.

            When called as :func:`erlab.io.load`, this argument defaults to the value
            set by :func:`erlab.io.set_data_dir` or :func:`erlab.io.loader_context`.
        single
            This argument is only used when :attr:`always_single
            <erlab.io.dataloader.LoaderBase.always_single>` is `False`, and `identifier`
            is given as a string or path-like object.

            If `identifier` points to a file that is included in a multiple file scan,
            the default behavior when `single` is `False` is to return data from all
            files in the same scan. How the data is combined is determined by the
            `combine` argument. If `True`, only the data from the file given is
            returned.
        combine
            Whether to attempt to combine multiple files into a single data object. If
            `False`, a list of data is returned. If `True`, the loader tries to combined
            the data into a single data object and return it. Depending on the type of
            each data object, the returned object can be a `xarray.DataArray`,
            `xarray.Dataset`, or a `xarray.DataTree`.

            This argument is only used when `single` is `False`.
        parallel
            Whether to load multiple files in parallel using the `joblib` library. For
            possible values, see :meth:`load_multiple_parallel
            <erlab.io.dataloader.LoaderBase.load_multiple_parallel>`.

            This argument is only used when `single` is `False`.
        progress
            Whether to show a progress bar when loading multiple files.

            This argument is only used when `single` is `False`.
        load_kwargs
            Additional keyword arguments to be passed to :meth:`load_single
            <erlab.io.dataloader.LoaderBase.load_single>`.
        **kwargs
            Additional keyword arguments are passed to :meth:`identify
            <erlab.io.dataloader.LoaderBase.identify>`.

        Returns
        -------
        `xarray.DataArray` or `xarray.Dataset` or `xarray.DataTree`
            The loaded data.

        Notes
        -----
        - The `data_dir` set by :func:`erlab.io.set_data_dir` or
          :func:`erlab.io.loader_context` is only used when called as
          :func:`erlab.io.load`. When called directly on a loader instance, the
          `data_dir` argument must be specified.
        - For convenience, the `data_dir` set by :func:`erlab.io.set_data_dir` or
          :func:`erlab.io.loader_context` is silently ignored when *all* of the
          following are satisfied:

          - `identifier` is an absolute path to an existing file.
          - `data_dir` is not explicitly provided.
          - The path created by joining `data_dir` and `identifier` does not point to an
            existing file.

          This way, absolute file paths can be passed directly to the loader without
          changing the default data directory. For instance, consider the following
          directory structure.

          .. code-block:: none

            cwd/
            ├── data/
            └── example.txt

          The following code will load ``./example.txt`` instead of raising an error
          that ``./data/example.txt`` is missing:

          .. code-block:: python

            import erlab

            erlab.io.set_data_dir("data")

            erlab.io.load("example.txt")

          However, if ``./data/example.txt`` also exists, the same code will load that
          one instead while warning about the ambiguity. This behavior may lead to
          unexpected results when the directory structure is not organized. Keep this in
          mind and try to keep all data files in the same level.

        """
        if self.always_single:
            single = True
        if load_kwargs is None:
            load_kwargs = {}

        if isinstance(identifier, int):
            if data_dir is None:
                raise ValueError(
                    "data_dir must be specified when identifier is an integer"
                )
            file_paths, coord_dict = typing.cast(
                "tuple[list[str], dict[str, Sequence]]",
                self.identify(identifier, data_dir, **kwargs),
            )  # Return type enforced by metaclass, cast to avoid mypy error
            # file_paths: list of file paths with at least one element

            if len(file_paths) == 1 and len(coord_dict) == 0:
                # Single file resolved
                data: xr.DataArray | xr.Dataset | xr.DataTree = self.load_single(
                    file_paths[0], **load_kwargs
                )
            else:
                # Multiple files resolved
                if combine:
                    data = self._combine_multiple(
                        self.load_multiple_parallel(
                            file_paths,
                            parallel=parallel,
                            progress=progress,
                            **load_kwargs,
                        ),
                        coord_dict,
                    )
                else:
                    return self.load_multiple_parallel(
                        file_paths,
                        parallel=parallel,
                        progress=progress,
                        post_process=True,
                        **load_kwargs,
                    )

        else:
            if data_dir is not None:
                # Generate full path to file
                identifier = os.path.join(data_dir, identifier)

                if not os.path.exists(identifier):
                    raise FileNotFoundError(
                        errno.ENOENT, os.strerror(errno.ENOENT), identifier
                    )

            if not single:
                # Get file name without extension and path
                basename_no_ext: str = os.path.splitext(os.path.basename(identifier))[0]

                # Infer index from file name
                new_identifier, additional_kwargs = self.infer_index(basename_no_ext)

                if new_identifier is not None:
                    # On success, load with the index
                    new_dir: str = os.path.dirname(identifier)

                    new_kwargs = kwargs | additional_kwargs
                    new_kwargs.setdefault("single", single)
                    new_kwargs.setdefault("combine", combine)
                    new_kwargs.setdefault("parallel", parallel)
                    new_kwargs.setdefault("load_kwargs", load_kwargs)
                    try:
                        return self.load(new_identifier, new_dir, **new_kwargs)
                    except Exception as e:
                        warning_message = (
                            f"Loading {basename_no_ext} with inferred index "
                            f"{new_identifier} resulted in an error:\n"
                            f"{type(e).__name__}: {e}\n"
                            "Possible causes:\n"
                            "- The inferred index may be incorrect.\n"
                            "- The file may be corrupted or in an unsupported format.\n"
                            "The data will be loaded as a single file."
                        )
                        erlab.utils.misc.emit_user_level_warning(warning_message)

                # On failure, assume single file
                single = True

            data = self.load_single(identifier, **load_kwargs)

        data = self.post_process_general(data)

        if not self.skip_validate:
            self.validate(data)

        return data

    @contextlib.contextmanager
    def extend_loader(
        self,
        *,
        name_map: dict[str, str | Iterable[str]] | None = None,
        coordinate_attrs: tuple[str, ...] | None = None,
        average_attrs: tuple[str, ...] | None = None,
        additional_attrs: dict[str, str | float | Callable[[xr.DataArray], str | float]]
        | None = None,
        overridden_attrs: tuple[str, ...] | None = None,
        additional_coords: dict[
            str, str | float | Callable[[xr.DataArray], str | float]
        ]
        | None = None,
        overridden_coords: tuple[str, ...] | None = None,
    ) -> Iterator[typing.Self]:
        """Context manager that temporarily extends various loader attributes.

        This context manager can be used to temporarily customize the behavior of the
        data loader. This is particularly useful when loading data across multiple
        files, where the :attr:`coordinate_attrs
        <erlab.io.dataloader.LoaderBase.coordinate_attrs>` can be extended to include
        additional information in the output data.

        Parameters
        ----------
        name_map
            Extends :attr:`name_map <erlab.io.dataloader.LoaderBase.name_map>`.
        coordinate_attrs
            Extends :attr:`coordinate_attrs
            <erlab.io.dataloader.LoaderBase.coordinate_attrs>`.
        average_attrs
            Extends :attr:`average_attrs
            <erlab.io.dataloader.LoaderBase.average_attrs>`.
        additional_attrs
            Extends :attr:`additional_attrs
            <erlab.io.dataloader.LoaderBase.additional_attrs>`.
        overridden_attrs
            Extends :attr:`overridden_attrs
            <erlab.io.dataloader.LoaderBase.overridden_attrs>`.
        additional_coords
            Extends :attr:`additional_coords
            <erlab.io.dataloader.LoaderBase.additional_coords>`.
        overridden_coords
            Extends :attr:`overridden_coords
            <erlab.io.dataloader.LoaderBase.overridden_coords>`.

        Example
        -------
        .. code-block:: python

            import erlab

            erlab.io.set_loader("loader_name")

            with erlab.io.extend_loader(coordinate_attrs=("scan_number",)):
                data = erlab.io.load("file_name")

        See Also
        --------
        :attr:`coordinate_attrs <erlab.io.dataloader.LoaderBase.coordinate_attrs>`
            The attribute that is temporarily extended.
        """
        old_vals: dict[str, typing.Any] = {}
        for attr, extend in {
            "name_map": name_map,
            "coordinate_attrs": coordinate_attrs,
            "average_attrs": average_attrs,
            "additional_attrs": additional_attrs,
            "overridden_attrs": overridden_attrs,
            "additional_coords": additional_coords,
            "overridden_coords": overridden_coords,
        }.items():
            if extend is not None:
                old_val = getattr(self, attr)
                old_vals[attr] = old_val
                if isinstance(extend, dict):
                    new_val = old_val.copy() | extend
                else:
                    new_val = old_val + tuple(extend)
                setattr(self, attr, new_val)
        try:
            yield self
        finally:
            for attr, old_val in old_vals.items():
                setattr(self, attr, old_val)

    def summarize(
        self,
        data_dir: str | os.PathLike,
        exclude: str | Sequence[str] | None = None,
        *,
        cache: bool = True,
        display: bool = True,
        rc: dict[str, typing.Any] | None = None,
    ) -> pandas.DataFrame | pandas.io.formats.style.Styler | None:
        """Summarize the data in the given directory.

        .. note::

            This method is not meant to be overriden in subclasses.

        Takes a path to a directory and summarizes the data in the directory to a table,
        much like a log file. This is useful for quickly inspecting the contents of a
        directory.

        The dataframe is formatted using the style from :meth:`get_styler
        <erlab.io.dataloader.LoaderBase.get_styler>` and displayed in the IPython shell.
        Results are cached in a pickle file in the directory.

        Parameters
        ----------
        data_dir
            Directory to summarize.
        exclude
            A string or sequence of strings specifying glob patterns for files to be
            excluded from the summary. If provided, caching will be disabled.
        cache
            Whether to use caching for the summary.
        display
            Whether to display the formatted dataframe using the IPython shell. If
            `False`, the dataframe will be returned without formatting. If `True` but
            the IPython shell is not detected, the dataframe styler will be returned.
        rc
            Optional dictionary of matplotlib rcParams to override the default for the
            plot in the interactive summary. Plot options such as the figure size and
            colormap can be changed using this argument.

        Returns
        -------
        pandas.DataFrame or pandas.io.formats.style.Styler or None
            Summary of the data in the directory.

            - If `display` is `False`, the summary DataFrame is returned.

            - If `display` is `True` and the IPython shell is detected, the summary will
              be displayed, and `None` will be returned.

              * If `ipywidgets` is installed, an interactive widget will be returned
                instead of `None`.

            - If `display` is `True` but the IPython shell is not detected, the styler
              for the summary DataFrame will be returned.

        """
        data_dir = pathlib.Path(data_dir)

        if not data_dir.is_dir():
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), str(data_dir)
            )

        pkl_path = data_dir / ".summary.pkl"
        df = None

        if exclude is not None:
            cache = False

        if pkl_path.is_file() and cache:
            try:
                df = pandas.read_pickle(pkl_path)
            except Exception:
                df = None

        if df is not None:
            contents = {str(f.relative_to(data_dir)) for f in data_dir.glob("[!.]*")}
            if contents != df.attrs.get("__contents", set()):
                # Cache is outdated
                df = None

        if df is None:
            df = self._generate_summary(data_dir, exclude)
            if cache and os.access(data_dir, os.W_OK):
                df.to_pickle(pkl_path)

        if not display:
            return df

        styled = self.get_styler(df)

        if erlab.utils.misc.is_interactive():
            import IPython.display

            with pandas.option_context(
                "display.max_rows", len(df), "display.max_columns", len(df.columns)
            ):
                IPython.display.display(styled)

            if importlib.util.find_spec("ipywidgets"):
                return self._isummarize(df, rc=rc)

            return None

        return styled

    def get_formatted_attr_or_coord(
        self,
        data: xr.DataArray,
        attr_or_coord_name: str | Callable[[xr.DataArray], typing.Any],
    ) -> typing.Any:
        """Return the formatted value of the given attribute or coordinate.

        The value is formatted using the function specified in :attr:`formatters
        <erlab.io.dataloader.LoaderBase.formatters>`. If the name is not found, an empty
        string is returned.

        Parameters
        ----------
        data : DataArray
            The data to extract the attribute or coordinate from.
        attr_or_coord_name : str or callable
            The name of the attribute or coordinate to extract. If a callable is passed,
            it is called with the data as the only argument.

        """
        if callable(attr_or_coord_name):
            return attr_or_coord_name(data)

        func = self.formatters.get(attr_or_coord_name, lambda x: x)

        if attr_or_coord_name in data.attrs:
            val = func(data.attrs[attr_or_coord_name])
        elif attr_or_coord_name in data.coords:
            val = data.coords[attr_or_coord_name].values
            if val.size == 1:
                val = func(val.item())
            else:
                val = np.array(list(map(func, val)), dtype=val.dtype)
        else:
            val = ""
        return val

    def _generate_summary(
        self, data_dir: str | os.PathLike, exclude: str | Sequence[str] | None = None
    ) -> pandas.DataFrame:
        """Generate a dataframe summarizing the data in the given directory.

        Parameters
        ----------
        data_dir
            Path to a directory.
        exclude
            A string or sequence of strings specifying glob patterns for files to be
            excluded from the summary.

        Returns
        -------
        pandas.DataFrame
            Summary of the data in the directory.

        """
        data_dir = pathlib.Path(data_dir)

        excluded: list[pathlib.Path] = []

        if exclude is not None:
            if isinstance(exclude, str):
                exclude = [exclude]

            for pattern in exclude:
                excluded = excluded + list(data_dir.glob(pattern))

        target_files: list[pathlib.Path] = [
            pathlib.Path(f)
            for f in self.files_for_summary(data_dir)
            if pathlib.Path(f) not in excluded
        ]
        target_files.sort()

        if not self.always_single:
            signatures: list[int | None] = [
                self.infer_index(f.stem)[0] for f in target_files
            ]

            # Removing duplicates that exist in same multi-file scan
            seen = set()  # set to track seen elements
            target_files_new = []
            for f, sig in zip(target_files, signatures, strict=True):
                if sig is not None and sig in seen:
                    # sig[0] == None for files that cannot be inferred, keep them
                    continue
                seen.add(sig)
                target_files_new.append(f)
            target_files = target_files_new

        columns = ["File Name", "Path", *self.summary_attrs.keys()]
        content = []

        def _add_content(
            data: xr.DataArray | xr.Dataset | xr.DataTree,
            file_path: pathlib.Path,
            suffix: str | None = None,
        ) -> None:
            if suffix is None:
                suffix = ""

            if isinstance(data, xr.DataArray):
                name = file_path.stem
                if suffix != "":
                    name = f"{name} ({suffix})"
                content.append(
                    [
                        file_path.stem,
                        str(file_path),
                        *(
                            self.get_formatted_attr_or_coord(data, v)
                            for v in self.summary_attrs.values()
                        ),
                    ]
                )

            elif isinstance(data, xr.Dataset):
                if len(data.data_vars) == 1:
                    _add_content(next(iter(data.data_vars.values())), file_path, suffix)
                else:
                    for k, darr in data.data_vars.items():
                        _add_content(darr, file_path, suffix=suffix + k)

            elif isinstance(data, xr.DataTree):
                for leaf in data.leaves:
                    _add_content(leaf.dataset, file_path, suffix=leaf.path)

        for f in target_files:
            try:
                _add_content(
                    typing.cast(
                        "xr.DataArray | xr.Dataset | xr.DataTree",
                        self.load(f, load_kwargs={"without_values": True}),
                    ),
                    f,
                )
            except Exception as e:
                erlab.utils.misc.emit_user_level_warning(
                    f"Failed to load {f} for summary: {e}"
                )

        sort_by = self.summary_sort if self.summary_sort is not None else "File Name"

        df = (
            pandas.DataFrame(content, columns=columns)
            .sort_values(sort_by)
            .set_index("File Name")
        )

        # Cache directory contents for determining whether cache is up-to-date
        contents = {str(f.relative_to(data_dir)) for f in data_dir.glob("[!.]*")}
        df.attrs["__contents"] = contents

        return df

    def _isummarize(
        self,
        summary: pandas.DataFrame | None = None,
        rc: dict[str, typing.Any] | None = None,
        **kwargs,
    ):
        rc_dict: dict[str, typing.Any] = {} if rc is None else rc

        if not importlib.util.find_spec("ipywidgets"):
            raise ImportError(
                "ipywidgets and IPython is required for interactive summaries"
            )

        if summary is None:
            kwargs["display"] = False
            df = typing.cast("pandas.DataFrame", self.summarize(**kwargs))
        else:
            df = summary

        import matplotlib.pyplot as plt
        from ipywidgets import (
            HTML,
            Button,
            Dropdown,
            FloatSlider,
            HBox,
            Layout,
            Output,
            Select,
            VBox,
        )
        from ipywidgets.widgets.interaction import show_inline_matplotlib_plots

        try:
            from erlab.interactive.imagetool.manager import is_running, show_in_manager

        except ImportError:
            manager_running: bool = False
        else:
            manager_running = is_running()

        # Temporary variable to store loaded data
        self._temp_data: xr.DataArray | None = None
        # !TODO: properly GC this variable

        def _format_data_info(series: pandas.Series) -> str:
            # Format data info as HTML table
            table = ""
            table += (
                "<div class='widget-inline-hbox widget-select' "
                "style='height:300px;overflow-y:auto;'>"
            )
            table += "<table class='widget-select'>"
            table += "<tbody>"

            for k, v in series.items():
                if k == "Path":
                    continue
                table += "<tr>"
                table += f"<td style='text-align:left;'><b>{k}</b></td>"
                table += f"<td style='text-align:left;'>{self.value_to_string(v)}</td>"
                table += "</tr>"

            table += "</tbody></table>"
            table += "</div>"
            return table

        def _update_data(
            _, *, full: bool = False, ret: bool = False
        ) -> None | xr.DataArray | xr.Dataset:
            # Load data for selected row
            series = df.loc[data_select.value]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                _path = pathlib.Path(series["Path"])

                full_button.disabled = True

                if not self.always_single:
                    idx, _ = self.infer_index(_path.stem)
                    if idx is not None:
                        ident = self.identify(idx, _path.parents[0])
                        if ident is not None:
                            n_scans = len(ident[0])
                            if n_scans > 1 and not full:
                                full_button.disabled = False

                out = self.load(_path, single=not full)
                if ret:
                    if isinstance(out, xr.DataArray):
                        return out.rename(_path.stem)
                    if isinstance(out, xr.Dataset):
                        return out
                    raise ValueError("Unsupported data type for itool")

                if isinstance(out, xr.DataArray):
                    self._temp_data = out
                del out

                data_info.value = _format_data_info(series)
            if self._temp_data is None:
                return None

            if self._temp_data.ndim == 4:
                # If the data is 4D, average over the last dimension, making it 3D
                self._temp_data = self._temp_data.mean(str(self._temp_data.dims[-1]))
                # !TODO: Add 2 sliders for 4D data

            if self._temp_data.ndim == 3:
                old_dim = str(dim_sel.value)

                dim_sel.unobserve(_update_sliders, "value")
                coord_sel.unobserve(_update_plot, "value")

                dim_sel.options = self._temp_data.dims
                # Set the default dimension to the one with the smallest size if
                # previous dimension is not present
                if old_dim in dim_sel.options:
                    dim_sel.value = old_dim
                else:
                    dim_sel.value = self._temp_data.dims[
                        np.argmin(self._temp_data.shape)
                    ]

                coord_sel.observe(_update_plot, "value")
                dim_sel.observe(_update_sliders, "value")

                dim_sel.disabled = False
                dim_sel.layout.visibility = "visible"
                coord_sel.disabled = False
                coord_sel.layout.visibility = "visible"

                _update_sliders(None)

            else:
                # 2D or 1D data, disable and hide dimension selection
                dim_sel.disabled = True
                dim_sel.layout.visibility = "hidden"
                coord_sel.disabled = True
                coord_sel.layout.visibility = "hidden"

            _update_plot(None)
            return None

        def _update_sliders(_) -> None:
            if out.block:
                return
            if self._temp_data is None:
                return

            scan_coords = self._temp_data[dim_sel.value].values

            dim_sel.unobserve(_update_sliders, "value")
            coord_sel.unobserve(_update_plot, "value")

            coord_sel.step = abs(scan_coords[1] - scan_coords[0])
            coord_sel.max = 1e100  # To ensure max > min before setting bounds
            coord_sel.min = scan_coords.min()
            coord_sel.max = scan_coords.max()

            coord_sel.observe(_update_plot, "value")
            dim_sel.observe(_update_sliders, "value")

        def _update_plot(_) -> None:
            out.clear_output(wait=True)
            if self._temp_data is None:
                return
            if not coord_sel.disabled:
                plot_data = self._temp_data.qsel({dim_sel.value: coord_sel.value})
            else:
                plot_data = self._temp_data

            old_rc = {k: v for k, v in plt.rcParams.items() if k in rc_dict}
            with out:
                plt.rcParams.update(rc_dict)
                plot_data.qplot(ax=plt.gca())
                plt.title("")  # Remove automatically generated title

                # Add line at Fermi level if the data is 2D and has an energy
                # dimension that includes zero
                if (plot_data.ndim == 2 and "eV" in plot_data.dims) and (
                    plot_data["eV"].values[0] * plot_data["eV"].values[-1] < 0
                ):
                    erlab.plotting.fermiline(
                        orientation="h" if plot_data.dims[0] == "eV" else "v"
                    )
                show_inline_matplotlib_plots()
            plt.rcParams.update(old_rc)

        def _next(_) -> None:
            # Select next row
            idx = list(df.index).index(data_select.value)
            if idx + 1 < len(df.index):
                data_select.value = list(df.index)[idx + 1]

        def _prev(_) -> None:
            # Select previous row
            idx = list(df.index).index(data_select.value)
            if idx - 1 >= 0:
                data_select.value = list(df.index)[idx - 1]

        # Buttons for navigation and loading full data
        prev_button = Button(description="Prev", layout=Layout(width="50px"))
        next_button = Button(description="Next", layout=Layout(width="50px"))
        full_button = Button(description="Load full", layout=Layout(width="100px"))
        itool_button = Button(description="itool", layout=Layout(width="50px"))
        prev_button.on_click(_prev)
        next_button.on_click(_next)
        full_button.on_click(lambda _: _update_data(None, full=True))
        if self.always_single:
            buttons = [prev_button, next_button]
        else:
            buttons = [prev_button, next_button, full_button]

        if manager_running:
            itool_button.on_click(
                lambda _: show_in_manager(
                    typing.cast(
                        "xr.DataArray | xr.Dataset",
                        _update_data(None, full=True, ret=True),
                    )
                )
            )
            buttons = [*buttons, itool_button]

        # List of data files
        data_select = Select(
            options=list(df.index), value=next(iter(df.index)), rows=10
        )
        data_select.observe(_update_data, "value")

        # HTML table for data info
        data_info = HTML()

        # Dropdown and slider for selecting dimension and coord
        dim_sel = Dropdown()
        dim_sel.observe(_update_sliders, "value")
        coord_sel = FloatSlider(continuous_update=True, readout_format=".3f")
        coord_sel.observe(_update_plot, "value")

        # Make UI
        ui = VBox([HBox(buttons), data_select, data_info, dim_sel, coord_sel])
        out = Output()
        out.block = False

        _update_data(None)

        return HBox(
            [ui, out],
            layout=Layout(
                display="grid",
                grid_template_columns="auto auto",
                grid_template_rows="auto",
            ),
        )

    def load_single(
        self, file_path: str | os.PathLike, *, without_values: bool = False
    ) -> xr.DataArray | xr.Dataset | xr.DataTree:
        r"""Load a single file and return it as an xarray data structure.

        All scan-specific postprocessing should be implemented in this method.

        This method must be implemented to return the *smallest possible data structure*
        that represents the data in a single file. For instance, if a single file
        contains a single scan region, the method should return a single
        `xarray.DataArray`. If it contains multiple regions, the method should return a
        `xarray.Dataset` or `xarray.DataTree` depending on whether the regions can be
        merged with without conflicts (i.e., all mutual coordinates of the regions are
        the same).

        Parameters
        ----------
        file_path
            Full path to the file to be loaded.
        without_values
            Used when creating a summary table. With this option set to `True`, only the
            coordinates and attributes of the output data are accessed so that the
            values can be replaced with placeholder numbers, speeding up the summary
            generation for lazy loading enabled file formats like HDF5 or NeXus.

        Returns
        -------
        DataArray or Dataset or DataTree
            The loaded data.

        Notes
        -----
        - For loaders with :attr:`always_single
          <erlab.io.dataloader.LoaderBase.always_single>` set to `False`, the return
          type of this method must be consistent across all associated files, i.e., for
          all files that can be returned together from :meth:`identify
          <erlab.io.dataloader.LoaderBase.identify>` so that they can be combined
          without conflicts. This should not be a problem in most cases since the data
          structure of associated files acquired during the same scan will be identical.
        - For `xarray.DataTree` objects, returned trees must be named with a unique
          identifier to avoid conflicts when combining.
        """
        raise NotImplementedError("method must be implemented in the subclass")

    def identify(
        self, num: int, data_dir: str | os.PathLike
    ) -> (
        tuple[
            list[pathlib.Path] | list[str], dict[str, Sequence] | dict[str, npt.NDArray]
        ]
        | None
    ):
        r"""Identify the files and coordinates for a given scan number.

        This method takes a scan index and transforms it into a list of file paths and
        coordinates. See below for the expected behavior.

        If no files are found for the given parameters, an empty list and an empty
        dictionary should be returned. Alternatively, return a single `None` to indicate
        a failure to identify the scan.

        Parameters
        ----------
        num
            The index of the scan to identify.
        data_dir
            The directory containing the data.

        Returns
        -------
        files : list of str or path-like
            A list of file paths.

            - For scans spread over multiple files, the list must contain all files that
              correspond to the given scan index.

            - For single file scans, the behavior differs based on the value of
              :attr:`always_single <erlab.io.dataloader.LoaderBase.always_single>`.

              - If `True`, all files that match the given scan index should be returned.
                In this case, only the first file will be loaded, and a warning will be
                shown to the user.

              - If `False`, there is no way to tell whether the returned files are part
                of a valid multiple-file scan. Hence, it is up to the loader to ensure
                that only a single file is returned and appropriate warnings are issued
                for single file scans when multiple files for a single scan are
                detected. See the source code of
                :meth:`erlab.io.plugins.merlin.MERLINLoader.identify` for an example.

        coord_dict : dict of str to sequence
            A dictionary mapping scan axes names to scan coordinates.

            The keys must match the coordinate name conventions used by the data
            returned by :meth:`load_single
            <erlab.io.dataloader.LoaderBase.load_single>`.

            - For scans spread over multiple files, the coordinates will be sequences,
              with each element corresponding to each file in ``files``.

            - For single file scans or multiple file scans that have no well-defined
              scan axes (such as multi-region scans), an empty dictionary should be
              returned.

        """
        raise NotImplementedError(
            "This loader does not support loading data by scan index. "
            "Try providing a file path instead."
        )

    def infer_index(self, name: str) -> tuple[int | None, dict[str, typing.Any]]:
        """Infer the index for the given file name.

        This method takes a file name with the path and extension stripped, and tries to
        infer the scan index from it. If the index can be inferred, it is returned along
        with additional keyword arguments that should be passed to :meth:`load
        <erlab.io.dataloader.LoaderBase.load>`. If the index is not found, `None` should
        be returned for the index, and an empty dictionary for additional keyword
        arguments.

        Parameters
        ----------
        name
            The base name of the file without the path and extension.

        Returns
        -------
        index
            The inferred index if found, otherwise None.
        additional_kwargs
            Additional keyword arguments to be passed to :meth:`identify
            <erlab.io.dataloader.LoaderBase.identify>` when the index is found. This
            argument is useful when the index alone is not enough to load the data.

        Note
        ----
        For loaders with :attr:`always_single
        <erlab.io.dataloader.LoaderBase.always_single>` set to `True`, this method is
        unused.

        """
        raise NotImplementedError("method must be implemented in the subclass")

    def files_for_summary(self, data_dir: str | os.PathLike) -> list[str | os.PathLike]:
        """Return a list of files that can be loaded by the loader.

        This method is used to select files that can be loaded by the loader when
        generating a summary.

        Parameters
        ----------
        data_dir
            The directory containing the data.

        Returns
        -------
        list of str or path-like
            A list of files that can be loaded by the loader.

        """
        raise NotImplementedError(
            f"loader '{self.name}' does not support folder summaries"
        )

    def combine_attrs(
        self,
        variable_attrs: Sequence[dict[str, typing.Any]],
        context: xr.Context | None = None,
    ) -> dict[str, typing.Any]:
        """Combine multiple attributes into a single attribute.

        This method is used as the ``combine_attrs`` argument in :func:`xarray.concat`
        and :func:`xarray.merge` when combining data from multiple files into a single
        object. By default, it has the same behavior as specifying
        `combine_attrs='override'` by taking the first set of attributes.

        The method can be overridden to provide fine-grained control over how the
        attributes are combined, e.g., by merging dictionaries or taking the average of
        some attributes.

        Parameters
        ----------
        variable_attrs
            A sequence of attributes to be combined.
        context
            The context in which the attributes are being combined. This has no effect,
            but is required by xarray.

        Returns
        -------
        dict[str, typing.Any]
            The combined attributes.

        """
        return dict(variable_attrs[0])

    @typing.overload
    def _combine_multiple(
        self, data_list: list[xr.DataArray], coord_dict: dict[str, Sequence]
    ) -> xr.DataArray: ...

    @typing.overload
    def _combine_multiple(
        self, data_list: list[xr.Dataset], coord_dict: dict[str, Sequence]
    ) -> xr.Dataset: ...

    @typing.overload
    def _combine_multiple(
        self, data_list: list[xr.DataTree], coord_dict: dict[str, Sequence]
    ) -> xr.DataTree: ...

    def _combine_multiple(
        self,
        data_list: list[xr.DataArray] | list[xr.Dataset] | list[xr.DataTree],
        coord_dict: dict[str, Sequence],
    ) -> xr.DataArray | xr.Dataset | xr.DataTree:
        if erlab.utils.misc.is_sequence_of(data_list, xr.DataTree):
            raise NotImplementedError(
                "Combining DataTrees into a single tree will be supported "
                "in a future release of ERLabPy. In the meantime, consider supplying "
                "`combine=False` to get a list of the data in each file, or "
                "`single=True` to load only one file."
            )

        if len(coord_dict) == 0:
            # No coordinates to combine given
            # Multiregion scans over multiple files may be provided like this

            if erlab.utils.misc.is_sequence_of(data_list, xr.DataTree):
                pass
            else:
                try:
                    return xr.combine_by_coords(
                        typing.cast(
                            "Sequence[xr.DataArray] | Sequence[xr.Dataset]", data_list
                        ),
                        combine_attrs=self.combine_attrs,
                        data_vars="all",
                        join="exact",
                    )
                except Exception as e:
                    raise RuntimeError(
                        "Failed to combine data. Try passing "
                        "`combine=False` to `erlab.io.load`"
                    ) from e

        if erlab.utils.misc.is_sequence_of(
            data_list, xr.DataArray
        ) or erlab.utils.misc.is_sequence_of(data_list, xr.Dataset):
            # If all coordinates are monotonic, all points are unique; in this case,
            # indexing along only the first dimension will not discard any data. For
            # example, hv-dependent cuts with coords 'hv' and 'beta' should be combined
            # into a 3D array rather than 4D, but position dependent cuts with coords
            # 'x' and 'y' should be combined into a 4D array. To handle the former case,
            # we combine along the first dimension only if all coordinates are
            # monotonic. Otherwise, we combine along all coordinates.
            if (
                len(coord_dict) > 1
                and len(data_list) > 1
                and all(
                    erlab.utils.array.is_monotonic(np.asarray(v), strict=True)
                    for v in coord_dict.values()
                )
            ):
                # We rename with process_keys after assigning coords and expanding dims.
                # This is necessary to ensure that coordinate_attrs are preserved.
                concat_dim = next(iter(coord_dict.keys()))
                concat_coord = coord_dict.pop(concat_dim)
                processed: list[xr.DataArray] = [
                    self.process_keys(
                        data.assign_coords({concat_dim: concat_coord[i]})
                        .expand_dims(concat_dim)
                        .assign_coords(
                            {k: (concat_dim, [v[i]]) for k, v in coord_dict.items()}
                        )
                    )
                    for i, data in enumerate(data_list)
                ]
            else:
                processed = [
                    self.process_keys(
                        data.assign_coords(
                            {k: v[i] for k, v in coord_dict.items()}
                        ).expand_dims(tuple(coord_dict.keys()))
                    )
                    for i, data in enumerate(data_list)
                ]

            # Magically combine the data
            combined = xr.combine_by_coords(
                processed,
                combine_attrs=self.combine_attrs,
                data_vars="all",
                join="exact",
            )

            if (
                isinstance(combined, xr.Dataset)
                and len(combined.data_vars) == 1
                and combined.attrs == {}
            ):
                # Named DataArrays combined into a Dataset, extract the DataArray
                var_name = next(iter(combined.data_vars))
                combined = combined[var_name]
                if combined.name is None or combined.name == "":
                    combined = combined.rename(var_name)

            return combined

        raise TypeError("input type must be homogeneous")

    def process_keys(
        self, data: xr.DataArray, key_mapping: dict[str, str] | None = None
    ) -> xr.DataArray:
        """Rename coordinates and attributes based on the given mapping.

        This method is used to rename coordinates and attributes. This method is called
        by :meth:`post_process <erlab.io.dataloader.LoaderBase.post_process>`. Extend or
        override this method to customize the renaming behavior.

        Parameters
        ----------
        data
            The data to be processed.
        key_mapping
            A dictionary mapping **original** names to **new** names. If not provided,
            :attr:`name_map_reversed <erlab.io.dataloader.LoaderBase.name_map_reversed>`
            is used.

        """
        if key_mapping is None:
            key_mapping = self._name_map_reversed

        # Rename coordinates
        data = data.rename({k: v for k, v in key_mapping.items() if k in data.coords})

        # For attributes, keep original attribute and add new with renamed keys
        new_attrs = {}
        for old_key, value in dict(data.attrs).items():
            if old_key in key_mapping:
                new_key = key_mapping[old_key]
                if (
                    new_key in (set(self.coordinate_attrs) | set(self.average_attrs))
                    and new_key in data.coords
                ):
                    # Renamed attribute is already a coordinate, remove
                    del data.attrs[old_key]
                else:
                    new_attrs[new_key] = value
        data = data.assign_attrs(new_attrs)

        # Move from attrs to coordinate if coordinate is not found
        return data.assign_coords(
            {
                a: data.attrs.pop(a)
                for a in (set(self.coordinate_attrs) | set(self.average_attrs))
                if a in data.attrs and a not in data.coords
            }
        )

    def post_process(self, darr: xr.DataArray) -> xr.DataArray:
        """Post-process the given `DataArray`.

        This method takes a single `DataArray` and applies post-processing steps such as
        renaming coordinates and attributes.

        This method is called by :meth:`post_process_general
        <erlab.io.dataloader.LoaderBase.post_process_general>`.

        Parameters
        ----------
        darr
            The `DataArray` to be post-processed.

        Returns
        -------
        DataArray
            The post-processed `DataArray`.

        Note
        ----
        When introducing a custom post-processing step in a loader, make sure to call
        the parent method in the subclass implementation.

        """
        darr = self.process_keys(darr)

        for k in self.average_attrs:
            if k in darr.coords:
                v = darr[k].values.mean()
                darr = darr.drop_vars(k).assign_attrs({k: v})

        new_attrs: dict[str, str | float] = {}
        for k, v in self.additional_attrs.items():
            if k not in darr.attrs:
                if callable(v):
                    new_attrs[k] = v(darr)
                else:
                    new_attrs[k] = v

        new_attrs = {
            k: v
            for k, v in self.additional_attrs.items()
            if k not in darr.attrs or k in self.overridden_attrs
        }
        new_attrs["data_loader_name"] = str(self.name)
        darr = darr.assign_attrs(new_attrs)

        new_coords = {}
        for k, v in self.additional_coords.items():
            if k not in darr.coords or k in self.overridden_coords:
                if callable(v):
                    new_coords[k] = v(darr)
                else:
                    new_coords[k] = v

        return darr.assign_coords(new_coords)

    def _reorder_coords(self, darr: xr.DataArray):
        """Sort the coordinates of the given DataArray."""
        return erlab.utils.array.sort_coord_order(
            darr,
            keys=itertools.chain(
                self.name_map.keys(),
                (s for s in self.coordinate_attrs if s not in self.name_map),
                self.additional_coords.keys(),
            ),
            dims_first=True,
        )

    @typing.overload
    def post_process_general(self, data: xr.DataArray) -> xr.DataArray: ...

    @typing.overload
    def post_process_general(self, data: xr.Dataset) -> xr.Dataset: ...

    @typing.overload
    def post_process_general(self, data: xr.DataTree) -> xr.DataTree: ...

    def post_process_general(
        self, data: xr.DataArray | xr.Dataset | xr.DataTree
    ) -> xr.DataArray | xr.Dataset | xr.DataTree:
        """Post-process any data structure.

        This method extends :meth:`post_process
        <erlab.io.dataloader.LoaderBase.post_process>` to handle any data structure.

        This method is called by :meth:`load <erlab.io.dataloader.LoaderBase.load>` as
        the final step in the data loading process.

        Parameters
        ----------
        data : DataArray or Dataset or DataTree
            The data to be post-processed.

            - If a `DataArray`, the data is post-processed using :meth:`post_process
              <erlab.io.dataloader.LoaderBase.post_process>`.
            - If a `Dataset`, a new `Dataset` containing each data variable
              post-processed using :meth:`post_process
              <erlab.io.dataloader.LoaderBase.post_process>` is returned. The attributes
              of the original `Dataset` are preserved.
            - If a `xarray.DataTree`, the post-processing is applied to each leaf node
              `Dataset`.

        Returns
        -------
        DataArray or Dataset or DataTree
            The post-processed data with the same type as the input.
        """
        if isinstance(data, xr.DataArray):
            return self._reorder_coords(self.post_process(data))

        if isinstance(data, xr.Dataset):
            return xr.Dataset(
                {
                    k: self._reorder_coords(self.post_process(v))
                    for k, v in data.data_vars.items()
                },
                attrs=data.attrs,
            )

        if isinstance(data, xr.DataTree):
            return typing.cast(
                "xr.DataTree", data.map_over_datasets(self.post_process_general)
            )

        raise TypeError(
            "data must be a DataArray, Dataset, or DataTree, but got " + type(data)
        )

    @classmethod
    def validate(cls, data: xr.DataArray | xr.Dataset | xr.DataTree) -> None:
        """Validate the input data to ensure it is in the correct format.

        Checks for the presence of all coordinates and attributes required for common
        analysis procedures like momentum conversion. If the data does not pass
        validation, a `ValidationError` is raised or a warning is issued, depending on
        the :attr:`strict_validation <erlab.io.dataloader.LoaderBase.strict_validation>`
        flag. Validation is skipped for loaders with :attr:`skip_validate
        <erlab.io.dataloader.LoaderBase.skip_validate>` set to `True`.

        Parameters
        ----------
        data : DataArray or Dataset or DataTree
            The data to be validated. If a `xarray.Dataset` or `xarray.DataTree` is
            passed, validation is performed on each data variable recursively.

        """
        if isinstance(data, xr.Dataset):
            for v in data.data_vars.values():
                cls.validate(v)
            return

        if isinstance(data, xr.DataTree):
            data.map_over_datasets(cls.validate)
            return

        for c in ("beta", "delta", "xi", "hv"):
            if c not in data.coords:
                cls._raise_or_warn(f"Missing coordinate '{c}'")

        if data.qinfo.get_value("sample_temp") is None:
            cls._raise_or_warn("Missing attribute 'sample_temp'")

        if "configuration" not in data.attrs:
            cls._raise_or_warn("Missing attribute 'configuration'")
            return

        if data.attrs["configuration"] not in (1, 2):
            if data.attrs["configuration"] not in (3, 4):
                cls._raise_or_warn(
                    f"Invalid configuration {data.attrs['configuration']}"
                )
            elif "chi" not in data.coords:
                cls._raise_or_warn("Missing coordinate 'chi'")

    def load_multiple_parallel(
        self,
        file_paths: list[str],
        *,
        parallel: bool | None = None,
        progress: bool = True,
        post_process: bool = False,
        **kwargs,
    ) -> list[xr.DataArray] | list[xr.Dataset] | list[xr.DataTree]:
        """Load from multiple files in parallel.

        Parameters
        ----------
        file_paths
            A list of file paths to load.
        parallel
            Whether to load data in parallel using `joblib`.

            - If `None`, parallel loading is enabled only if the number of files is
              greater than the loader's :attr:`parallel_threshold
              <erlab.io.dataloader.LoaderBase.parallel_threshold>`.

            - If `True`, data loading will always be performed in parallel.

            - If `False`, data will be loaded sequentially.
        progress
            Whether to show a progress bar.
        post_process
            Whether to post-process each data object after loading.
        **kwargs
            Additional keyword arguments to be passed to :meth:`load_single
            <erlab.io.dataloader.LoaderBase.load_single>`.

        Returns
        -------
        A list of the loaded data.
        """
        if parallel is None:
            parallel = len(file_paths) > self.parallel_threshold

        if post_process:

            def _load_func(filename):
                return self.load(filename, single=True, load_kwargs=kwargs)

        else:

            def _load_func(filename):
                return self.load_single(filename, **kwargs)

        tqdm_kw = {
            "desc": "Loading",
            "total": len(file_paths),
            "disable": not progress,
        }

        if parallel:
            with erlab.utils.parallel.joblib_progress(**tqdm_kw) as _:
                return joblib.Parallel(n_jobs=-1, max_nbytes=None)(
                    joblib.delayed(_load_func)(f) for f in file_paths
                )

        return [_load_func(f) for f in tqdm.tqdm(file_paths, **tqdm_kw)]

    @classmethod
    def _raise_or_warn(cls, msg: str) -> None:
        if cls.strict_validation:
            raise ValidationError(msg)
        erlab.utils.misc.emit_user_level_warning(msg, ValidationWarning)


class _RegistryBase:
    """Base class for the loader registry.

    This class implements the singleton pattern, ensuring that only one instance of the
    registry is created and used throughout the application.
    """

    __instance: _RegistryBase | None = None

    def __new__(cls):
        if not isinstance(cls.__instance, cls):
            cls.__instance = super().__new__(cls)
        return cls.__instance

    @classmethod
    def instance(cls) -> typing.Self:
        """Return the registry instance."""
        return cls()


class LoaderRegistry(_RegistryBase):
    """Registry of loader plugins.

    Stores and manages data loaders. The loaders can be accessed by name or alias in a
    dictionary-like manner.

    Most public methods of this class instance can be accessed through the `erlab.io`
    namespace.

    """

    _loaders: typing.ClassVar[dict[str, LoaderBase | type[LoaderBase]]] = {}
    """Mapping of registered loaders."""

    _alias_mapping: typing.ClassVar[dict[str, str]] = {}
    """Mapping of aliases to loader names."""

    _current_loader: LoaderBase | None = None
    _current_data_dir: pathlib.Path | None = None

    @property
    def current_loader(self) -> LoaderBase | None:
        """Current loader."""
        return self._current_loader

    @current_loader.setter
    def current_loader(self, loader: str | LoaderBase | None) -> None:
        self.set_loader(loader)

    @property
    def current_data_dir(self) -> os.PathLike | None:
        """Directory to search for data files."""
        return self._current_data_dir

    @current_data_dir.setter
    def current_data_dir(self, data_dir: str | os.PathLike | None) -> None:
        self.set_data_dir(data_dir)

    @property
    def default_data_dir(self) -> os.PathLike | None:
        """Deprecated alias for current_data_dir.

        .. deprecated:: 3.0.0

            Use :attr:`current_data_dir` instead.
        """
        warnings.warn(
            "`default_data_dir` is deprecated, use `current_data_dir` instead",
            FutureWarning,
            stacklevel=1,
        )
        return self.current_data_dir

    def _register(self, loader_class: type[LoaderBase]) -> None:
        # Add class to loader
        self._loaders[loader_class.name] = loader_class

        # Add aliases to mapping
        self._alias_mapping[loader_class.name] = loader_class.name
        if loader_class.aliases is not None:
            if not loader_class.__module__.startswith("erlab.io.plugins"):
                warnings.warn(
                    "Loader aliases are deprecated. Users are encouraged to use the "
                    "name of the loader instead.",
                    FutureWarning,
                    stacklevel=1,
                )
            for alias in loader_class.aliases:
                self._alias_mapping[alias] = loader_class.name

    def keys(self) -> KeysView[str]:
        return self._loaders.keys()

    def items(self) -> ItemsView[str, LoaderBase | type[LoaderBase]]:
        return self._loaders.items()

    def get(self, key: str) -> LoaderBase:
        """Get a loader instance by name or alias."""
        loader_name = self._alias_mapping.get(key)
        if loader_name is None:
            raise LoaderNotFoundError(key)

        loader = self._loaders.get(loader_name)
        if key != loader_name:
            erlab.utils.misc.emit_user_level_warning(
                "Loader aliases are deprecated. Access the loader with the loader name "
                f"'{loader_name}' instead.",
                FutureWarning,
            )

        if loader is None:
            raise LoaderNotFoundError(key)

        if not isinstance(loader, LoaderBase):
            # If not an instance, create one
            loader = loader()
            self._loaders[loader_name] = loader

        return loader

    def __iter__(self) -> Iterator[str]:
        return iter(self._loaders)

    def __getitem__(self, key: str) -> LoaderBase:
        return self.get(key)

    def __getattr__(self, key: str) -> LoaderBase:
        try:
            return self.get(key)
        except LoaderNotFoundError as e:
            raise AttributeError(str(e)) from e

    def set_loader(self, loader: str | LoaderBase | None) -> None:
        """Set the current data loader.

        All subsequent calls to `load` will use the provided loader.

        Parameters
        ----------
        loader
            The loader to set. It can be either a string representing the name or alias
            of the loader, or a valid loader class.

        Example
        -------

        >>> erlab.io.set_loader("merlin")
        >>> dat_merlin_1 = erlab.io.load(...)
        >>> dat_merlin_2 = erlab.io.load(...)

        """
        if isinstance(loader, str):
            self._current_loader = self.get(loader)
        else:
            self._current_loader = loader

    @contextlib.contextmanager
    def loader_context(
        self, loader: str | None = None, data_dir: str | os.PathLike | None = None
    ) -> Iterator[LoaderBase]:
        """
        Context manager for the current data loader and data directory.

        Parameters
        ----------
        loader : str, optional
            The name or alias of the loader to use in the context.
        data_dir : str or os.PathLike, optional
            The data directory to use in the context.

        Examples
        --------
        - Load data within a context manager:

          >>> with erlab.io.loader_context("merlin"):
          ...     dat_merlin = erlab.io.load(...)

        - Load data with different loaders and directories:

          >>> erlab.io.set_loader("ssrl52", data_dir="/path/to/dir1")
          >>> dat_ssrl_1 = erlab.io.load(...)
          >>> with erlab.io.loader_context("merlin", data_dir="/path/to/dir2"):
          ...     dat_merlin = erlab.io.load(...)
          >>> dat_ssrl_2 = erlab.io.load(...)

        """
        if loader is None and data_dir is None:
            raise ValueError(
                "At least one of loader or data_dir must be specified in the context"
            )

        if loader is not None:
            old_loader: LoaderBase | None = self.current_loader
            self.set_loader(loader)

        if data_dir is not None:
            old_data_dir = self.current_data_dir
            self.set_data_dir(data_dir)

        try:
            yield typing.cast("LoaderBase", self.current_loader)
        finally:
            if loader is not None:
                self.set_loader(old_loader)

            if data_dir is not None:
                self.set_data_dir(old_data_dir)

    def set_data_dir(self, data_dir: str | os.PathLike | None) -> None:
        """Set the default data directory for the data loader.

        All subsequent calls to :func:`erlab.io.load` will use the provided `data_dir`
        unless specified.

        Parameters
        ----------
        data_dir
            The default data directory to use.

        Note
        ----
        This will only affect :func:`erlab.io.load`. If the loader's ``load`` method is
        called directly, it will not use the default data directory.

        """
        if data_dir is None:
            self._current_data_dir = None
            return

        self._current_data_dir = pathlib.Path(data_dir).resolve(strict=True)

    def load(
        self,
        identifier: str | os.PathLike | int,
        data_dir: str | os.PathLike | None = None,
        *,
        single: bool = False,
        combine: bool = True,
        parallel: bool = False,
        progress: bool = True,
        load_kwargs: dict[str, typing.Any] | None = None,
        **kwargs,
    ) -> (
        xr.DataArray
        | xr.Dataset
        | xr.DataTree
        | list[xr.DataArray]
        | list[xr.Dataset]
        | list[xr.DataTree]
    ):
        loader, default_dir = self._get_current_defaults()

        if (
            default_dir is not None
            and data_dir is None
            and not isinstance(identifier, int)
            and os.path.exists(identifier)
        ):
            abs_file = pathlib.Path(identifier).resolve()
            default_file = (default_dir / identifier).resolve()

            if default_file.exists() and abs_file != default_file:
                erlab.utils.misc.emit_user_level_warning(
                    f"Found {identifier!s} in the default directory "
                    f"{default_dir!s}, but conflicting file {abs_file!s} was found. "
                    "The first file will be loaded. "
                    "Consider specifying the directory explicitly."
                )
            else:
                # If the identifier is a path to a file, ignore default_dir
                default_dir = None

        if data_dir is None:
            data_dir = default_dir

        return loader.load(
            identifier,
            data_dir=data_dir,
            single=single,
            combine=combine,
            parallel=parallel,
            progress=progress,
            load_kwargs=load_kwargs,
            **kwargs,
        )

    def extend_loader(
        self,
        name_map: dict[str, str | Iterable[str]] | None = None,
        coordinate_attrs: tuple[str, ...] | None = None,
        average_attrs: tuple[str, ...] | None = None,
        additional_attrs: dict[str, str | float | Callable[[xr.DataArray], str | float]]
        | None = None,
        overridden_attrs: tuple[str, ...] | None = None,
        additional_coords: dict[str, str | int | float] | None = None,
        overridden_coords: tuple[str, ...] | None = None,
    ) -> Iterator[LoaderBase]:
        loader, _ = self._get_current_defaults()
        return loader.extend_loader(
            name_map=name_map,
            coordinate_attrs=coordinate_attrs,
            average_attrs=average_attrs,
            additional_attrs=additional_attrs,
            overridden_attrs=overridden_attrs,
            additional_coords=additional_coords,
            overridden_coords=overridden_coords,
        )

    def summarize(
        self,
        data_dir: str | os.PathLike | None = None,
        exclude: str | Sequence[str] | None = None,
        *,
        cache: bool = True,
        display: bool = True,
        rc: dict[str, typing.Any] | None = None,
    ) -> pandas.DataFrame | pandas.io.formats.style.Styler | None:
        loader, default_dir = self._get_current_defaults()

        if data_dir is None:
            data_dir = default_dir

        return loader.summarize(
            data_dir=data_dir, exclude=exclude, cache=cache, display=display, rc=rc
        )

    def _get_current_defaults(self):
        if self.current_loader is None:
            raise ValueError(
                "No loader has been set. Set a loader with `erlab.io.set_loader` first"
            )
        return self.current_loader, self.current_data_dir

    def __repr__(self) -> str:
        # Store string variables used when calculating the max len
        descriptions = [
            v.description if hasattr(v, "description") else "No description"
            for v in self._loaders.values()
        ]
        class_names = [
            f"{type(v).__module__}.{type(v).__qualname__}"
            if isinstance(v, LoaderBase)
            else f"{v.__module__}.{v.__qualname__}"
            for v in self._loaders.values()
        ]

        # Calculate the maximum width for each column
        max_name_len = max(len(k) for k in self._loaders)
        max_desc_len = max(len(desc) for desc in descriptions)
        max_cls_len = max(len(cls_name) for cls_name in class_names)

        # Create the header row with dynamic padding
        header = (
            f"{'Name':<{max_name_len}} | "
            f"{'Description':<{max_desc_len}} | "
            f"{'Loader class':<{max_cls_len}}"
        )

        # Create the separator row
        separator = "-" * (max_name_len + max_desc_len + max_cls_len + 6)

        # Create the rows with dynamic padding
        rows = [header, separator]
        for k, desc, cls_name in zip(
            self._loaders.keys(), descriptions, class_names, strict=True
        ):
            rows.append(
                f"{k:<{max_name_len}} | "
                f"{desc:<{max_desc_len}} | "
                f"{cls_name:<{max_cls_len}}"
            )

        return "\n".join(rows)

    def _repr_html_(self) -> str:
        rows: list[tuple[str, str, str]] = [("Name", "Description", "Loader class")]

        for k, v in self._loaders.items():
            desc: str = v.description if hasattr(v, "description") else ""

            # May be either a class or an instance
            if isinstance(v, LoaderBase):
                v = type(v)

            cls_name = f"{v.__module__}.{v.__qualname__}"
            rows.append((k, desc, cls_name))

        return erlab.utils.formatting.format_html_table(rows, header_rows=1)

    load.__doc__ = LoaderBase.load.__doc__
    extend_loader.__doc__ = LoaderBase.extend_loader.__doc__
    summarize.__doc__ = LoaderBase.summarize.__doc__


loaders: LoaderRegistry = LoaderRegistry.instance()
"""Global instance of :class:`LoaderRegistry <erlab.io.dataloader.LoaderRegistry>`."""
