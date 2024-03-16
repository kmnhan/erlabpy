r"""Base functionality for implementing data loaders

This module provides a base class `LoaderBase` for implementing data loaders. Data
loaders are plugins used to load data from various file formats. Each data loader that
subclasses `LoaderBase` is registered on import in `loaders`.

Loaded ARPES data must contain several attributes and coordinates. See the
implementation of `LoaderBase.validate` for details.

For any data loader plugin subclassing `LoaderBase`, the following attributes and
methods must be defined: `LoaderBase.name`, `LoaderBase.aliases`,
`LoaderBase.rename_keys`, `LoaderBase.coordinate_attrs`, `LoaderBase.additional_attrs`,
`LoaderBase.always_single`, `LoaderBase.skip_validate`, :func:`LoaderBase.load_single`,
:func:`LoaderBase.identify`, :func:`LoaderBase.infer_index`, and
:func:`LoaderBase.generate_summary`.

If additional post-processing is required, the :func:`LoaderBase.post_process` method
can be extended to include the necessary functionality.

"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Sequence, Iterable

import joblib
import xarray as xr
import pandas as pd


class ValidationError(Exception):
    """This exception is raised when the loaded data fails validation checks."""

    pass


class LoaderBase:
    """Base class for all data loaders."""

    name: str | None = None
    """
    Name of the loader. Using a unique and descriptive name is recommended. For easy
    access, it is recommended to use a name that passes :func:`str.isidentifier`.
    """

    aliases: list[str] | None = None
    """List of alternative names for the loader."""

    rename_keys: dict[str, str] = {}
    """
    Dictionary mapping original coordinate or attribute names to new coordinate or
    attribute names.
    """

    coordinate_attrs: tuple[str, ...] = tuple()
    """
    Names of attributes (after renaming) that should be treated as coordinates.
    
    Note
    ----
    Although the data loader tries to preserve the original attributes, the attributes
    given here, both before and after renaming, will be removed from attrs for
    consistency.
    """

    additional_attrs: dict[str, str | int | float] = {}
    """Additional attributes to be added to the data."""

    always_single: bool = False
    """
    If `True`, this indicates that all individual scans always lead to a single data
    file. No concatenation of data from multiple files will be performed.
    """

    skip_validate: bool = False
    """If `True`, validation checks will be skipped."""

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if not hasattr(cls, "name"):
            raise NotImplementedError("name attribute must be defined in the subclass")

        if not cls.name.startswith("_"):
            LoaderRegistry.instance().register(cls)

    def load(
        self,
        identifier: str | os.PathLike | int | None,
        data_dir: str | os.PathLike | None = None,
        **kwargs: dict,
    ) -> xr.DataArray | xr.Dataset | list[xr.DataArray]:
        """Load ARPES data.

        Parameters
        ----------
        identifier
            Value that identifies a scan uniquely. If a string or path-like object is
            given, it is assumed to be the path to the data file. If an integer is
            given, it is assumed to be a number that specifies the scan number, and is
            used to automatically determine the path to the data file(s).
        data_dir
            Where to look for the data. If `None`, the default data directory will be
            used.
        single
            For some endstations, data for a single scan is saved over multiple files.
            This argument only has effect for such endstations. When a single file
            within a multiple file scan is provided to `identifier`, the default
            behavior when `single` is `False` is to infer the coordinates and return a
            single concatenated array that contains data from all files. If `single` is
            set to `True`, only the data from the file given is returned.
        **kwargs
            Additional keyword arguments are passed to `identify`.

        Returns
        -------
        xarray.DataArray or xarray.Dataset or list of xarray.DataArray
            The loaded data.

        """
        single = kwargs.pop("single", False)

        if self.always_single:
            single = True

        if isinstance(identifier, int):
            if data_dir is None:
                raise ValueError(
                    "data_dir must be specified when identifier is an integer"
                )
            file_paths, coord_dict = self.identify(identifier, data_dir, **kwargs)

            if len(file_paths) == 0:
                raise ValueError(
                    f"Failed to resolve identifier {identifier} "
                    f"for data directory {data_dir}"
                )
            elif len(file_paths) == 1:
                # Single file resolved
                data = self.load_single(file_paths[0])
            else:
                # Multiple files resolved
                data = self.combine_multiple(
                    self._load_multiple_parallel(file_paths), coord_dict
                )
        else:
            if data_dir is not None:
                # Generate full path to file
                identifier = os.path.join(data_dir, identifier)

            if not single:
                # Get file name without extension and path
                basename_no_ext: str = os.path.splitext(os.path.basename(identifier))[0]

                # Infer index from file name
                new_identifier: int | None = self.infer_index(basename_no_ext)

                if new_identifier is not None:
                    # On success, load with the index
                    new_dir: str = os.path.dirname(identifier)
                    return self.load(new_identifier, new_dir, single, **kwargs)
                else:
                    # On failure, assume single file
                    single = True

            data = self.load_single(identifier)

        data = self.post_process_general(data)

        if not self.skip_validate:
            self.validate(data)

        return data

    def summarize(
        self, data_dir: str | os.PathLike, usecache: bool = True, **kwargs
    ) -> pd.DataFrame:
        """
        Takes a path to a directory and summarizes the data in the directory to a pandas
        DataFrame, much like a log file. This is useful for quickly inspecting the
        contents of a directory.

        Results are cached in a pickle file in the directory. If the pickle file is not
        found, the summary is generated with `generate_summary` and cached.

        Parameters
        ----------
        data_dir
            Directory to summarize.
        usecache
            Whether to use the cached summary if available. If `False`, the summary will
            be regenerated and cached.
        **kwargs
            Additional keyword arguments to be passed to `generate_summary`.

        Returns
        -------
        pandas.DataFrame
            Summary of the data in the directory.

        """
        pkl_path = os.path.join(data_dir, ".summary.pkl")
        if usecache:
            try:
                return pd.read_pickle(pkl_path)
            except FileNotFoundError:
                pass
        summary = self.generate_summary(data_dir, **kwargs)
        summary.to_pickle(pkl_path)
        return summary

    def load_single(
        self, file_path: str | os.PathLike
    ) -> xr.DataArray | xr.Dataset | list[xr.DataArray]:
        """Load a single file and return it in applicable format.

        Any scan-specific postprocessing should be implemented in this method. When the
        single file contains many regions, the method should return a single dataset
        whenever the data can be merged with `xarray.merge` without conflicts.
        Otherwise, a list of `xarray.DataArray`\ s should be returned.

        Parameters
        ----------
        file_paths
            Full path to the file to be loaded.

        Returns
        -------
        xarray.DataArray or xarray.Dataset or list of xarray.DataArray
            The loaded data.

        """
        raise NotImplementedError("method must be implemented in the subclass")

    def identify(
        self, num: int, data_dir: str | os.PathLike
    ) -> tuple[list[str], dict[str, Iterable]]:
        """Identify the files and coordinates for a given scan number.

        This method takes a scan index and transforms it into a list of file paths and
        coordinates.

        Parameters
        ----------
        num
            The index of the scan to identify.
        data_dir
            The directory containing the data.

        Returns
        -------
        files : list[str]
            A list of file paths.
        coord_dict : dict[str, Iterable]
            A dictionary mapping scan axes names to scan coordinates. For scans spread
            over multiple files, the coordinates will be iterables corresponding to each
            file in the `files` list. For single file scans, an empty dictionary is
            returned.

        """
        raise NotImplementedError("method must be implemented in the subclass")

    def infer_index(self, name: str) -> int | None:
        """Infer the index for the given file name.

        Parameters
        ----------
        name
            The base name of the file without the path and extension.

        Returns
        -------
        index
            The inferred index if found, otherwise None.

        Note
        ----
        This method is used to determine all files for a given scan. Hence, for loaders
        with `always_single` set to `True`, this method does not have to be implemented.

        """
        raise NotImplementedError("method must be implemented in the subclass")

    def generate_summary(self, data_dir: str | os.PathLike) -> pd.DataFrame:
        """Takes a path to a directory and summarizes the data in the directory to a
        pandas DataFrame, much like a log file. This is useful for quickly inspecting
        the contents of a directory.

        Parameters
        ----------
        data_dir
            Path to a directory.

        Returns
        -------
        pandas.DataFrame
            Summary of the data in the directory.

        """
        raise NotImplementedError("This loader does not support folder summaries")

    def combine_multiple(
        self,
        data_list: list[xr.DataArray | xr.Dataset],
        coord_dict: dict[str, Sequence],
    ) -> xr.DataArray | xr.Dataset | Sequence[xr.DataArray | xr.Dataset]:

        if len(coord_dict) == 0:
            try:
                # Try to merge the data without conflicts
                return xr.merge(data_list)
            except:
                # On failure, return a list
                return data_list
        else:
            for i in range(len(data_list)):
                data_list[i] = data_list[i].assign_coords(
                    {k: v[i] for k, v in coord_dict.items()}
                )
            return xr.concat(
                data_list, dim=list(coord_dict.keys())[0], coords="different"
            )

    def post_process_general(
        self, data: xr.DataArray | xr.Dataset | list[xr.DataArray | xr.Dataset]
    ) -> xr.DataArray | xr.Dataset | list[xr.DataArray | xr.Dataset]:
        if isinstance(data, xr.DataArray):
            return self.post_process(data)

        elif isinstance(data, list):
            return [self.post_process(d) for d in data]

        elif isinstance(data, xr.Dataset):
            for k, v in data.data_vars.items():
                data[k] = self.post_process(v)
            return data

    def process_keys(
        self, data: xr.DataArray, key_mapping: dict[str, str] | None = None
    ) -> xr.DataArray:
        if key_mapping is None:
            key_mapping = self.rename_keys

        # Rename coordinates
        data = data.rename({k: v for k, v in key_mapping.items() if k in data.coords})

        # For attributes, keep original attribute and add new with renamed keys
        new_attrs = {}
        for k, v in dict(data.attrs).items():
            if k in key_mapping:
                new_key = key_mapping[k]
                if new_key in self.coordinate_attrs and new_key in data.coords:
                    # Renamed attribute is already a coordinate, remove
                    del data.attrs[k]
                else:
                    new_attrs[new_key] = v
        data = data.assign_attrs(new_attrs)

        # Move from attrs to coordinate if coordinate is not found
        data = data.assign_coords(
            {
                a: data.attrs.pop(a)
                for a in self.coordinate_attrs
                if a in data.attrs and a not in data.coords
            }
        )
        return data

    def post_process(self, data: xr.DataArray) -> xr.DataArray:
        data = self.process_keys(data)
        data = data.assign_attrs(self.additional_attrs)
        return data

    @classmethod
    def validate(cls, data: xr.DataArray | xr.Dataset):
        """Validate the input data to ensure it is in the correct format.

        Checks for the presence of all required coordinates and attributes. If the data
        does not pass validation, a `ValidationError` is raised. Validation is skipped
        for loaders with attribute `skip_validate` set to `True`.

        Parameters
        ----------
        data
            The data to be validated.

        Raises
        ------
        ValidationError

        """
        if isinstance(data, list):
            for d in data:
                cls.validate(d)
            return

        if isinstance(data, xr.Dataset):
            for v in data.data_vars.values():
                cls.validate(v)
            return

        for c in ("alpha", "beta", "delta", "xi", "hv"):
            if c not in data.coords:
                raise ValidationError(f"Missing coordinate {c}")
                # print(f"Missing coordinate {c}")

        for a in ("configuration", "temp_sample"):
            if a not in data.attrs:
                raise ValidationError(f"Missing attribute {c}")
                # print(f"Missing attribute {a}")

        if data.attrs["configuration"] not in (1, 2):
            if data.attrs["configuration"] not in (3, 4):
                raise ValidationError(
                    f"Invalid configuration {data.attrs['configuration']}"
                )
                # print(f"Invalid configuration {data.attrs['configuration']}")
            else:
                # Configuration with deflector, check for chi coordinate
                if "chi" not in data.coords:
                    raise ValidationError("Missing coordinate chi")
                    # print("Missing coordinate chi")

    def _load_multiple_parallel(
        self, file_paths: list[str]
    ) -> list[xr.DataArray | xr.Dataset]:
        if len(file_paths) < 15:
            n_jobs = 1
        else:
            n_jobs = -1

        return joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(self.load_single)(f) for f in file_paths
        )


class RegistryBase:
    """Base class for the loader registry.

    This class implements the singleton pattern, ensuring that only one instance of the
    registry is created and used throughout the application.
    """

    __instance = None

    def __new__(cls):
        if not isinstance(cls.__instance, cls):
            cls.__instance = super(RegistryBase, cls).__new__(cls)
        return cls.__instance

    @classmethod
    def instance(cls) -> LoaderRegistry:
        """Returns the registry instance."""
        return cls()


class LoaderRegistry(RegistryBase):
    loaders: dict[str, LoaderBase] = dict()
    """Registered loaders \n\n:meta hide-value:"""

    alias_mapping: dict[str, str] = dict()
    """Mapping of aliases to loader names \n\n:meta hide-value:"""

    current_loader: LoaderBase | None = None
    """Current loader \n\n:meta hide-value:"""

    default_data_dir: str | os.PathLike | None = None
    """Default directory to search for data files \n\n:meta hide-value:"""

    def register(self, loader_class: type[LoaderBase]):
        # Add class to loader
        self.loaders[loader_class.name] = loader_class

        # Add aliases to mapping
        self.alias_mapping[loader_class.name] = loader_class.name
        for alias in loader_class.aliases:
            self.alias_mapping[alias] = loader_class.name

    def get(self, key: str) -> type[LoaderBase]:
        try:
            loader_name = self.alias_mapping.get(key)
            loader = self.loaders.get(loader_name)

            if type(loader) == type:
                # If not an instance, create one
                self.loaders[key] = loader()
                loader = self.loaders[key]

            return loader

        except KeyError:
            raise ValueError(f"Loader for {key} not found")

    def __getitem__(self, key: str) -> type[LoaderBase]:
        return self.get(key)

    def set_loader(self, loader: str | LoaderBase):
        """Set the current data loader.

        All subsequent calls to `load` will use the loader set here.

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
            self.current_loader = self.get(loader)
        else:
            self.current_loader = loader

    @contextlib.contextmanager
    def loader_context(self, loader: str):
        """Context manager for temporarily changing the current data loader.

        The loader set here will only be used within the context manager.

        Parameters
        ----------
        loader
            The new loader to set as the current loader.

        Examples
        --------
        - Load data within a context manager:

          >>> with erlab.io.loader_context("merlin"):
          ...     dat_merlin = erlab.io.load(...)

        - Load data with different loaders:

          >>> erlab.io.set_loader("ssrl52")
          >>> dat_ssrl_1 = erlab.io.load(...)
          >>> with erlab.io.loader_context("merlin"):
          ...     dat_merlin = erlab.io.load(...)
          >>> dat_ssrl_2 = erlab.io.load(...)

        """
        old_loader: LoaderBase = self.current_loader
        self.set_loader(loader)
        try:
            yield self.current_loader
        finally:
            self.set_loader(old_loader)

    def set_data_dir(self, data_dir: str | os.PathLike):
        """Set the default data directory for the data loader.

        All subsequent calls to `load` will use the `data_dir` set here unless
        specified.

        Parameters
        ----------
        data_dir
            The path to a directory.

        Note
        ----
        This will only affect `load`. If the loader's ``load`` method is called
        directly, it will not use the default data directory.

        """
        self.default_data_dir = data_dir

    def load(
        self,
        identifier: str | os.PathLike | int | None,
        data_dir: str | os.PathLike | None = None,
        **kwargs: dict,
    ) -> xr.DataArray | xr.Dataset | list[xr.DataArray]:

        loader, default_dir = self._get_current_defaults()

        if data_dir is None:
            data_dir = default_dir

        if not isinstance(identifier, int) and os.path.isfile(identifier):
            # If the identifier is a path to a file, ignore data_dir
            data_dir = None

        return loader.load(identifier, data_dir=data_dir, **kwargs)

    def summarize(
        self, data_dir: str | os.PathLike, usecache: bool = True, **kwargs
    ) -> xr.DataArray | xr.Dataset | list[xr.DataArray]:

        loader, default_dir = self._get_current_defaults()

        if data_dir is None:
            data_dir = default_dir

        return loader.summarize(data_dir, usecache, **kwargs)

    def _get_current_defaults(self):
        if self.current_loader is None:
            raise ValueError(
                "No loader has been set. Set a loader with `erlab.io.set_loader` first"
            )
        return self.current_loader, self.default_data_dir

    def __repr__(self) -> str:
        out = "Registered data loaders\n====================\n\n"
        out += "Loaders\n-------\n" + "\n".join(
            [f"{k}: {v}" for k, v in self.loaders.items()]
        )
        out += "\n\n"
        out += "Aliases\n-------\n" + "\n".join(
            [
                f"{k}: {v.aliases}"
                for k, v in self.loaders.items()
                if v.aliases is not None
            ]
        )
        return out

    def _repr_html_(self) -> str:
        out = "<h3>Registered data loaders</h3>"
        out += "<table>"
        out += (
            "<tr>"
            "<th style='text-align:left;'><b>Name</b></th>"
            "<th style='text-align:left;'><b>Aliases</b></th>"
            "<th style='text-align:left;'><b>Loader class</b></th>"
            "</tr>"
        )
        for k, v in self.loaders.items():
            aliases = ", ".join(v.aliases) if v.aliases is not None else ""

            # May be either a class or an instance
            if type(v) != type:
                v = type(v)

            cls_name = f"{v.__module__}.{v.__qualname__}"

            out += (
                "<tr>"
                f"<tr><td style='text-align:left;'>{k}</td>"
                f"<td style='text-align:left;'>{aliases}</td>"
                f"<td style='text-align:left;'>{cls_name}</td>"
                "</tr>"
            )
        out += "</table>"

        return out

    load.__doc__ = LoaderBase.load.__doc__
    summarize.__doc__ = LoaderBase.summarize.__doc__


loaders: LoaderRegistry = LoaderRegistry.instance()
"""Global instance of `LoaderRegistry`\n\n:meta hide-value:"""

set_loader = loaders.set_loader
loader_context = loaders.loader_context
set_data_dir = loaders.set_data_dir
load = loaders.load
summarize = loaders.summarize
