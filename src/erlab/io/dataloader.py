r"""Base functionality for implementing data loaders.

This module provides a base class `LoaderBase` for implementing data loaders. Data
loaders are plugins used to load data from various file formats. Each data loader that
subclasses `LoaderBase` is registered on import in `loaders`.

Loaded ARPES data must contain several attributes and coordinates. See the
implementation of `LoaderBase.validate` for details.

A detailed guide on how to implement a data loader can be found in
:doc:`../user-guide/io`.

If additional post-processing is required, the :func:`LoaderBase.post_process` method
can be extended to include the necessary functionality.

"""

from __future__ import annotations

import contextlib
import datetime
import importlib
import itertools
import os
import warnings
from typing import TYPE_CHECKING, Any, ClassVar, Self, cast

import joblib
import numpy as np
import pandas
import xarray as xr

from erlab.utils.array import is_monotonic, is_uniform_spaced

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        ItemsView,
        Iterable,
        KeysView,
        Mapping,
        Sequence,
    )

    DataFromSingleFile = xr.DataArray | xr.Dataset | list[xr.DataArray]


class ValidationError(Exception):
    """Raised when the loaded data fails validation checks."""


class ValidationWarning(UserWarning):
    """Issued when the loaded data fails validation checks."""


class LoaderNotFoundError(Exception):
    """Raised when a loader is not found in the registry."""

    def __init__(self, key: str):
        super().__init__(f"Loader for name or alias {key} not found in the registry")


class LoaderBase:
    """Base class for all data loaders."""

    name: str
    """
    Name of the loader. Using a unique and descriptive name is recommended. For easy
    access, it is recommended to use a name that passes :func:`str.isidentifier`.
    """

    aliases: Iterable[str] | None = None
    """List of alternative names for the loader."""

    name_map: ClassVar[dict[str, str | Iterable[str]]] = {}
    """
    Dictionary that maps **new** coordinate or attribute names to **original**
    coordinate or attribute names. If there are multiple possible names for a single
    attribute, the value can be passed as an iterable.
    """

    coordinate_attrs: tuple[str, ...] = ()
    """
    Names of attributes (after renaming) that should be treated as coordinates.

    Attributes mentioned here will be moved from attrs to coordinates. This means that
    it will be propagated when concatenating data from multiple files. If a listed
    attribute is not found, it will be silently skipped.

    Note
    ----
    Although the data loader tries to preserve the original attributes, the attributes
    given here, both before and after renaming, will be removed from attrs for
    consistency.
    """

    average_attrs: tuple[str, ...] = ()
    """
    Names of attributes or coordinates (after renaming) that should be averaged over.

    This is useful for attributes that may slightly vary between scans. If a listed
    attribute is not found, it will be silently skipped.

    Note
    ----
    The attributes are just converted to coordinates upon loading and are averaged in
    the post-processing step.
    """

    additional_attrs: ClassVar[dict[str, str | int | float]] = {}
    """Additional attributes to be added to the data after loading.

    If an attribute with the same name is already present, it will be skipped.
    """

    additional_coords: ClassVar[dict[str, str | int | float]] = {}
    """Additional non-dimension coordinates to be added to the data after loading."""

    always_single: bool = True
    """
    If `True`, this indicates that all individual scans always lead to a single data
    file. No concatenation of data from multiple files will be performed.
    """

    skip_validate: bool = False
    """If `True`, validation checks will be skipped."""

    strict_validation: bool = False
    """
    If `True`, validation check will raise a `ValidationError` on the first failure
    instead of warning. Useful for debugging data loaders.
    """

    @property
    def name_map_reversed(self) -> dict[str, str]:
        """A reversed version of the name_map dictionary.

        This property is useful for mapping original names to new names.

        """
        return self.reverse_mapping(self.name_map)

    @property
    def coordinate_and_average_attrs(self) -> tuple[str, ...]:
        """Return a tuple of coordinate and average attributes."""
        return self.coordinate_attrs + self.average_attrs

    @property
    def file_dialog_methods(self) -> dict[str, tuple[Callable, dict[str, Any]]]:
        """Map from file dialog names to the called method and its arguments.

        This property can be overridden specify the file dialog methods to be called
        from the load menu of the ImageTool GUI.

        Returns
        -------
        loader_mapping
            A dictionary mapping the file dialog names to the called method and its
            arguments. The method should be a callable that takes a single positional
            argument which is a path to a data file, for instance ``self.load``. The
            arguments should be a dictionary containing keyword arguments to be passed
            to the method. It can be left empty if no additional arguments are required.
        """
        return {}

    @staticmethod
    def reverse_mapping(mapping: Mapping[str, str | Iterable[str]]) -> dict[str, str]:
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
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if not hasattr(cls, "name"):
            raise NotImplementedError("name attribute must be defined in the subclass")

        if not cls.name.startswith("_"):
            LoaderRegistry.instance().register(cls)

    @classmethod
    def formatter(cls, val: object):
        """Format the given value based on its type.

        This method is used when formatting the cells of the summary dataframe.

        Parameters
        ----------
        val
            The value to be formatted.

        Returns
        -------
        str or object
            The formatted value.

        Note
        ----
        This function formats the given value based on its type. It supports formatting
        for various types including numpy arrays, lists of strings, floating-point
        numbers, integers, and datetime objects.

        The function also tries to replace the Unicode hyphen-minus sign "-" (U+002D)
        with the better-looking Unicode minus sign "−" (U+2212) in most cases.

        - For numpy arrays:
            - If the array has a size of 1, the value is recursively formatted using
              `formatter(val.item())`.
            - If the array can be squeezed to a 1-dimensional array, the following are
              applied.

                - If the array is evenly spaced, the start, end, step, and length values
                  are formatted and returned as a string in the format "start→end (step,
                  length)".
                - If the array is monotonic increasing or decreasing but not evenly
                  spaced, the start, end, and length values are formatted and returned
                  as a string in the format "start→end (length)".
                - If all elements are equal, the value is recursively formatted using
                  `formatter(val[0])`.
                - If the array is not monotonic, the minimum and maximum values are
                  formatted and returned as a string in the format "min~max".

            - For arrays with more dimensions, the array is returned as is.

        - For lists:
            The list is grouped by consecutive equal elements, and the count of each
            element is formatted and returned as a string in the format
            "[element]×count".

        - For floating-point numbers:
            - If the number is an integer, it is formatted as an integer using
              `formatter(np.int64(val))`.
            - Otherwise, it is formatted as a floating-point number with 4 decimal
              places and returned as a string.

        - For integers:
            The integer is returned as a string.

        - For datetime objects:
            The datetime object is formatted as a string in the format "%Y-%m-%d
            %H:%M:%S".

        - For other types:
            The value is returned as is.

        Examples
        --------
        >>> formatter(np.array([0.1, 0.15, 0.2]))
        '0.1→0.2 (0.05, 3)'

        >>> formatter(np.array([1.0, 2.0, 2.1]))
        '1→2.1 (3)'

        >>> formatter(np.array([1.0, 2.1, 2.0]))
        '1~2.1 (3)'

        >>> formatter([1, 1, 2, 2, 2, 3, 3, 3, 3])
        '[1]×2, [2]×3, [3]×4'

        >>> formatter(3.14159)
        '3.1416'

        >>> formatter(42.0)
        '42'

        >>> formatter(42)
        '42'

        >>> formatter(datetime.datetime(2024, 1, 1, 12, 0, 0, 0))
        '2024-01-01 12:00:00'
        """
        if isinstance(val, np.ndarray):
            if val.size == 1:
                return cls.formatter(val.item())

            elif val.squeeze().ndim == 1:
                val = val.squeeze()

                if is_uniform_spaced(val):
                    start, end, step = tuple(
                        cls.formatter(v) for v in (val[0], val[-1], val[1] - val[0])
                    )
                    return f"{start}→{end} ({step}, {len(val)})".replace("-", "−")

                elif is_monotonic(val):
                    if val[0] == val[-1]:
                        return cls.formatter(val[0])

                    return (
                        f"{cls.formatter(val[0])}→{cls.formatter(val[-1])} ({len(val)})"
                    )

                else:
                    mn, mx = tuple(cls.formatter(v) for v in (np.min(val), np.max(val)))
                    return f"{mn}~{mx} ({len(val)})"

            else:
                return val

        elif isinstance(val, list):
            return ", ".join(
                [f"[{k}]×{len(tuple(g))}" for k, g in itertools.groupby(val)]
            )

        elif np.issubdtype(type(val), np.floating):
            val = cast(np.floating, val)
            if val.is_integer():
                return cls.formatter(np.int64(val))
            else:
                return np.format_float_positional(val, precision=4, trim="-").replace(
                    "-", "−"
                )
        elif np.issubdtype(type(val), np.integer):
            return str(val).replace("-", "−")

        elif isinstance(val, datetime.datetime):
            return val.strftime("%Y-%m-%d %H:%M:%S")

        else:
            return val

    @classmethod
    def get_styler(cls, df: pandas.DataFrame) -> pandas.io.formats.style.Styler:
        """Return a styled version of the given dataframe.

        This method, along with `formatter`, determines the display formatting of the
        summary dataframe. Override this method to change the display style.

        Parameters
        ----------
        df
            Summary dataframe as returned by `generate_summary`.

        Returns
        -------
        pandas.io.formats.style.Styler
            The styler to be displayed.

        """
        style = df.style.format(cls.formatter)

        hidden = [c for c in ("Time", "Path") if c in df.columns]
        if len(hidden) > 0:
            style = style.hide(hidden, axis="columns")

        return style

    def load(
        self,
        identifier: str | int,
        data_dir: str | None = None,
        **kwargs,
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
            For some setups, data for a single scan is saved over multiple files. This
            argument is only used for such setups. When `identifier` is resolved to a
            single file within a multiple file scan, the default behavior when `single`
            is `False` is to return a single concatenated array that contains data from
            all files in the same scan. If `single` is set to `True`, only the data from
            the file given is returned. This argument is ignored when `identifier` is a
            number.
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
                    self.load_multiple_parallel(file_paths), coord_dict
                )
        else:
            if data_dir is not None:
                # Generate full path to file
                identifier = os.path.join(data_dir, identifier)

            if not single:
                # Get file name without extension and path
                basename_no_ext: str = os.path.splitext(os.path.basename(identifier))[0]

                # Infer index from file name
                new_identifier, additional_kwargs = self.infer_index(basename_no_ext)

                if new_identifier is not None:
                    # On success, load with the index
                    new_dir: str = os.path.dirname(identifier)

                    new_kwargs = kwargs | additional_kwargs
                    return self.load(
                        new_identifier, new_dir, single=single, **new_kwargs
                    )
                else:
                    # On failure, assume single file
                    single = True

            data = self.load_single(identifier)

        data = self.post_process_general(data)

        if not self.skip_validate:
            self.validate(data)

        return data

    def summarize(
        self,
        data_dir: str | os.PathLike,
        usecache: bool = True,
        *,
        cache: bool = True,
        display: bool = True,
        **kwargs,
    ) -> pandas.DataFrame | pandas.io.formats.style.Styler | None:
        """Summarize the data in the given directory.

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
        usecache
            Whether to use the cached summary if available. If `False`, the summary will
            be regenerated. The cache will be updated if `cache` is `True`.
        cache
            Whether to cache the summary in a pickle file in the directory. If `False`,
            no cache will be created or updated. Note that existing cache files will not
            be deleted, and will be used if `usecache` is `True`.
        display
            Whether to display the formatted dataframe using the IPython shell. If
            `False`, the dataframe will be returned without formatting. If `True` but
            the IPython shell is not detected, the dataframe styler will be returned.
        **kwargs
            Additional keyword arguments to be passed to `generate_summary`.

        Returns
        -------
        df : pandas.DataFrame or pandas.io.formats.style.Styler or None
            Summary of the data in the directory.

            - If `display` is `False`, the summary DataFrame is returned.

            - If `display` is `True` and the IPython shell is detected, the summary will
              be displayed, and `None` will be returned.

              * If `ipywidgets` is installed, an interactive widget will be returned
                instead of `None`.

            - If `display` is `True` but the IPython shell is not detected, the styler
              for the summary DataFrame will be returned.

        """
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Directory {data_dir} not found")

        pkl_path = os.path.join(data_dir, ".summary.pkl")
        df = None
        if usecache:
            try:
                df = pandas.read_pickle(pkl_path)
                df = df.head(len(df))
            except FileNotFoundError:
                pass

        if df is None:
            df = self.generate_summary(data_dir, **kwargs)
            if cache:
                try:
                    df.to_pickle(pkl_path)
                except OSError:
                    warnings.warn(
                        f"Failed to cache summary to {pkl_path}", stacklevel=1
                    )

        if not display:
            return df

        styled = self.get_styler(df)

        try:
            shell = get_ipython().__class__.__name__  # type: ignore[name-defined]
            if display and (
                shell in ["ZMQInteractiveShell", "TerminalInteractiveShell"]
            ):
                from IPython.display import display  # type: ignore[assignment]

                with pandas.option_context(
                    "display.max_rows", len(df), "display.max_columns", len(df.columns)
                ):
                    display(styled)  # type: ignore[misc]

                if importlib.util.find_spec("ipywidgets"):
                    return self._isummarize(df)

                return None

        except NameError:
            pass

        return styled

    def isummarize(self, df: pandas.DataFrame | None = None, **kwargs):
        """Display an interactive summary.

        This method provides an interactive summary of the data using ipywidgets and
        matplotlib.

        Parameters
        ----------
        df
            A summary dataframe as returned by `generate_summary`. If None, a dataframe
            will be generated using `summarize`. Defaults to None.
        **kwargs
            Additional keyword arguments to be passed to `summarize` if `df` is None.

        Note
        ----
        This method requires `ipywidgets` to be installed. If not found, an
        `ImportError` will be raised.

        """
        if not importlib.util.find_spec("ipywidgets"):
            raise ImportError(
                "ipywidgets and IPython is required for interactive summaries"
            )
        if df is None:
            kwargs["display"] = False
            df = cast(pandas.DataFrame, self.summarize(**kwargs))

        self._isummarize(df)

    def _isummarize(self, df: pandas.DataFrame):
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

        import erlab.plotting.erplot as eplt

        self._temp_data: xr.DataArray | None = None

        def _format_data_info(series) -> str:
            table = ""
            table += (
                "<div class='widget-inline-hbox widget-select' "
                "style='height:220px;overflow-y:auto;'>"
            )
            table += "<table class='widget-select'>"
            table += "<tbody>"

            for k, v in series.items():
                if k == "Path":
                    continue
                table += "<tr>"
                table += f"<td style='text-align:left;'><b>{k}</b></td>"
                table += f"<td style='text-align:left;'>{self.formatter(v)}</td>"
                table += "</tr>"

            table += "</tbody></table>"
            table += "</div>"
            return table

        def _update_data(_, *, full: bool = False):
            series = df.loc[data_select.value]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                path = series["Path"]

                full_button.disabled = True

                if not self.always_single:
                    idx, _ = self.infer_index(
                        os.path.splitext(os.path.basename(path))[0]
                    )
                    if idx is not None:
                        n_scans = len(self.identify(idx, os.path.dirname(path))[0])
                        if n_scans > 1 and not full:
                            full_button.disabled = False

                out = self.load(path, single=not full)
                if isinstance(out, xr.DataArray):
                    self._temp_data = out
                del out

                data_info.value = _format_data_info(series)
            if self._temp_data is None:
                return

            if self._temp_data.ndim == 4:
                # If the data is 4D, average over the last dimension, making it 3D
                self._temp_data = self._temp_data.mean(str(self._temp_data.dims[-1]))

            if self._temp_data.ndim == 3:
                dim_sel.unobserve(_update_sliders, "value")
                coord_sel.unobserve(_update_plot, "value")

                dim_sel.options = self._temp_data.dims
                # Set the default dimension to the one with the smallest size
                dim_sel.value = self._temp_data.dims[np.argmin(self._temp_data.shape)]

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

        def _update_sliders(_):
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

        def _update_plot(_):
            if self._temp_data is None:
                return
            if not coord_sel.disabled:
                plot_data = self._temp_data.qsel({dim_sel.value: coord_sel.value})
            else:
                plot_data = self._temp_data

            out.clear_output(wait=True)
            with out:
                plot_data.qplot(ax=plt.gca())
                plt.title("")  # Remove automatically generated title

                # Add line at Fermi level if the data is 2D and has an energy dimension
                if plot_data.ndim == 2 and "eV" in plot_data.dims:
                    # Check if binding
                    if plot_data["eV"].values[0] * plot_data["eV"].values[-1] < 0:
                        eplt.fermiline(
                            orientation="h" if plot_data.dims[0] == "eV" else "v"
                        )
                show_inline_matplotlib_plots()

        def _next(_):
            # Select next row
            idx = list(df.index).index(data_select.value)
            if idx + 1 < len(df.index):
                data_select.value = list(df.index)[idx + 1]

        def _prev(_):
            # Select previous row
            idx = list(df.index).index(data_select.value)
            if idx - 1 >= 0:
                data_select.value = list(df.index)[idx - 1]

        prev_button = Button(description="Prev", layout=Layout(width="50px"))
        prev_button.on_click(_prev)

        next_button = Button(description="Next", layout=Layout(width="50px"))
        next_button.on_click(_next)

        full_button = Button(description="Load full", layout=Layout(width="100px"))
        full_button.on_click(lambda _: _update_data(None, full=True))

        buttons = [prev_button, next_button]
        if not self.always_single:
            buttons.append(full_button)

        data_select = Select(options=list(df.index), value=next(iter(df.index)), rows=8)
        data_select.observe(_update_data, "value")

        data_info = HTML()

        dim_sel = Dropdown()
        dim_sel.observe(_update_sliders, "value")

        coord_sel = FloatSlider(continuous_update=True, readout_format=".3f")
        coord_sel.observe(_update_plot, "value")

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
        self, file_path: str | os.PathLike
    ) -> xr.DataArray | xr.Dataset | list[xr.DataArray]:
        r"""Load a single file and return it in applicable format.

        Any scan-specific postprocessing should be implemented in this method. When the
        single file contains many regions, the method should return a single dataset
        whenever the data can be merged with `xarray.merge` without conflicts.
        Otherwise, a list of `xarray.DataArray`\ s should be returned.

        Parameters
        ----------
        file_path
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
        coordinates. For scans spread over multiple files, the coordinates must be a
        dictionary mapping scan axes names to scan coordinates. For single file scans,
        the list should contain only one file path and coordinates must be an empty
        dictionary.

        The keys of the coordinates must be transformed to new names prior to returning
        by using the mapping returned by the `name_map_reversed` property.

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

    def infer_index(self, name: str) -> tuple[int | None, dict[str, Any]]:
        """Infer the index for the given file name.

        This method takes a file name with the path and extension stripped, and tries to
        infer the scan index from it. If the index can be inferred, it is returned along
        with additional keyword arguments that should be passed to `load`. If the index
        is not found, `None` should be returned for the index, and an empty dictionary
        for additional keyword arguments.

        Parameters
        ----------
        name
            The base name of the file without the path and extension.

        Returns
        -------
        index
            The inferred index if found, otherwise None.
        additional_kwargs
            Additional keyword arguments to be passed to `load` when the index is found.
            This argument is useful when the index alone is not enough to load the data.

        Note
        ----
        This method is used to determine all files for a given scan. Hence, for loaders
        with `always_single` set to `True`, this method does not have to be implemented.

        """
        raise NotImplementedError("method must be implemented in the subclass")

    def generate_summary(self, data_dir: str | os.PathLike) -> pandas.DataFrame:
        """Generate a dataframe summarizing the data in the given directory.

        Takes a path to a directory and summarizes the data in the directory to a pandas
        DataFrame, much like a log file. This is useful for quickly inspecting the
        contents of a directory.

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

    def combine_attrs(
        self,
        variable_attrs: Sequence[dict[str, Any]],
        context: xr.Context | None = None,
    ) -> dict[str, Any]:
        """Combine multiple attributes into a single attribute.

        This method is used as the ``combine_attrs`` argument in :func:`xarray.concat`
        and :func:`xarray.merge` when combining data from multiple files into a single
        object. By default, it has the same behavior as specifying
        `combine_attrs='override'` by taking the first set of attributes.

        The method can be overridden to provide fine-grained control over how the
        attributes are combined.

        Parameters
        ----------
        variable_attrs
            A sequence of attributes to be combined.
        context
            The context in which the attributes are being combined. This has no effect,
            but is required by xarray.

        Returns
        -------
        dict[str, Any]
            The combined attributes.

        """
        return dict(variable_attrs[0])

    def combine_multiple(
        self,
        data_list: list[xr.DataArray | xr.Dataset | list[xr.DataArray]],
        coord_dict: dict[str, Iterable],
    ) -> (
        xr.DataArray | xr.Dataset | list[xr.DataArray | xr.Dataset | list[xr.DataArray]]
    ):
        if len(coord_dict) == 0:
            try:
                # Try to merge the data without conflicts
                return xr.merge(data_list, combine_attrs=self.combine_attrs)
            except:  # noqa: E722
                # On failure, return a list
                return data_list
        else:
            for i in range(len(data_list)):
                if isinstance(data_list[i], list):
                    data_list[i] = self.combine_multiple(data_list[i], coord_dict={})

                if not isinstance(data_list[i], list):
                    data_list[i] = data_list[i].assign_coords(
                        {k: v[i] for k, v in coord_dict.items()}
                    )
            try:
                return xr.concat(
                    data_list,
                    dim=next(iter(coord_dict.keys())),
                    coords="different",
                    combine_attrs=self.combine_attrs,
                )
            except:  # noqa: E722
                return data_list

    def process_keys(
        self, data: xr.DataArray, key_mapping: dict[str, str] | None = None
    ) -> xr.DataArray:
        if key_mapping is None:
            key_mapping = self.name_map_reversed

        # Rename coordinates
        data = data.rename({k: v for k, v in key_mapping.items() if k in data.coords})

        # For attributes, keep original attribute and add new with renamed keys
        new_attrs = {}
        for k, v in dict(data.attrs).items():
            if k in key_mapping:
                new_key = key_mapping[k]
                if (
                    new_key in self.coordinate_and_average_attrs
                    and new_key in data.coords
                ):
                    # Renamed attribute is already a coordinate, remove
                    del data.attrs[k]
                else:
                    new_attrs[new_key] = v
        data = data.assign_attrs(new_attrs)

        # Move from attrs to coordinate if coordinate is not found
        data = data.assign_coords(
            {
                a: data.attrs.pop(a)
                for a in self.coordinate_and_average_attrs
                if a in data.attrs and a not in data.coords
            }
        )
        return data

    def post_process(self, darr: xr.DataArray) -> xr.DataArray:
        darr = self.process_keys(darr)

        for k in self.average_attrs:
            if k in darr.coords:
                v = darr[k].values.mean()
                darr = darr.drop_vars(k).assign_attrs({k: v})

        darr = darr.assign_attrs(
            {k: v for k, v in self.additional_attrs.items() if k not in darr.attrs}
            | {"data_loader_name": str(self.name)}
        )
        darr = darr.assign_coords(self.additional_coords)

        # Make coordinate order pretty
        new_coords = {}
        coord_dict = dict(darr.coords)
        for d in darr.dims:
            new_coords[d] = coord_dict.pop(d)
        for d in itertools.chain(self.name_map.keys(), self.additional_coords.keys()):
            if d in coord_dict:
                new_coords[d] = coord_dict.pop(d)
        new_coords = new_coords | coord_dict

        darr = xr.DataArray(
            darr.values, coords=new_coords, dims=darr.dims, attrs=darr.attrs
        )
        return darr

    def post_process_general(
        self, data: xr.DataArray | xr.Dataset | list[xr.DataArray]
    ) -> xr.DataArray | xr.Dataset | list[xr.DataArray]:
        if isinstance(data, xr.DataArray):
            return self.post_process(data)

        elif isinstance(data, list):
            return [self.post_process(d) for d in data]

        elif isinstance(data, xr.Dataset):
            return xr.Dataset(
                {k: self.post_process(v) for k, v in data.data_vars.items()},
                attrs=data.attrs,
            )

    @classmethod
    def validate(
        cls, data: xr.DataArray | xr.Dataset | list[xr.DataArray | xr.Dataset]
    ) -> None:
        """Validate the input data to ensure it is in the correct format.

        Checks for the presence of all required coordinates and attributes. If the data
        does not pass validation, a `ValidationError` is raised or a warning is issued,
        depending on the value of the `strict_validation` flag. Validation is skipped
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
                cls._raise_or_warn(f"Missing coordinate {c}")

        for a in ("configuration", "temp_sample"):
            if a not in data.attrs:
                cls._raise_or_warn(f"Missing attribute {a}")

        if data.attrs["configuration"] not in (1, 2):
            if data.attrs["configuration"] not in (3, 4):
                cls._raise_or_warn(
                    f"Invalid configuration {data.attrs['configuration']}"
                )
            elif "chi" not in data.coords:
                cls._raise_or_warn("Missing coordinate chi")

    def load_multiple_parallel(
        self, file_paths: list[str], n_jobs: int | None = None
    ) -> list[xr.DataArray | xr.Dataset | list[xr.DataArray]]:
        """Load multiple files in parallel.

        Parameters
        ----------
        file_paths
            A list of file paths to load.
        n_jobs
            The number of jobs to run in parallel. If `None`, the number of jobs is set
            to 1 for less than 15 files and to -1 (all CPU cores) for 15 or more files.

        Returns
        -------
        A list of the loaded data.
        """
        if n_jobs is None:
            if len(file_paths) < 15:
                n_jobs = 1
            else:
                n_jobs = -1

        return joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(self.load_single)(f) for f in file_paths
        )

    @classmethod
    def _raise_or_warn(cls, msg: str):
        if cls.strict_validation:
            raise ValidationError(msg)
        else:
            warnings.warn(msg, ValidationWarning, stacklevel=2)


class RegistryBase:
    """Base class for the loader registry.

    This class implements the singleton pattern, ensuring that only one instance of the
    registry is created and used throughout the application.
    """

    __instance: RegistryBase | None = None

    def __new__(cls):
        if not isinstance(cls.__instance, cls):
            cls.__instance = super().__new__(cls)
        return cls.__instance

    @classmethod
    def instance(cls) -> Self:
        """Return the registry instance."""
        return cls()


class LoaderRegistry(RegistryBase):
    loaders: ClassVar[dict[str, LoaderBase | type[LoaderBase]]] = {}
    """Registered loaders \n\n:meta hide-value:"""

    alias_mapping: ClassVar[dict[str, str]] = {}
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
        if loader_class.aliases is not None:
            for alias in loader_class.aliases:
                self.alias_mapping[alias] = loader_class.name

    def keys(self) -> KeysView[str]:
        return self.loaders.keys()

    def items(self) -> ItemsView[str, LoaderBase | type[LoaderBase]]:
        return self.loaders.items()

    def get(self, key: str) -> LoaderBase:
        loader_name = self.alias_mapping.get(key)
        if loader_name is None:
            raise LoaderNotFoundError(key)

        loader = self.loaders.get(loader_name)

        if loader is None:
            raise LoaderNotFoundError(key)

        if not isinstance(loader, LoaderBase):
            # If not an instance, create one
            loader = loader()
            self.loaders[loader_name] = loader

        return loader

    def __getitem__(self, key: str) -> LoaderBase:
        return self.get(key)

    def __getattr__(self, key: str) -> LoaderBase:
        try:
            return self.get(key)
        except LoaderNotFoundError as e:
            raise AttributeError(str(e)) from e

    def set_loader(self, loader: str | LoaderBase | None):
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
    def loader_context(
        self, loader: str | None = None, data_dir: str | os.PathLike | None = None
    ):
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
            old_data_dir = self.default_data_dir
            self.set_data_dir(data_dir)

        try:
            yield self.current_loader
        finally:
            if loader is not None:
                self.set_loader(old_loader)

            if data_dir is not None:
                self.set_data_dir(old_data_dir)

    def set_data_dir(self, data_dir: str | os.PathLike | None):
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
        if data_dir is not None and not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Directory {data_dir} not found")
        self.default_data_dir = data_dir

    def load(
        self,
        identifier: str | os.PathLike | int | None,
        data_dir: str | os.PathLike | None = None,
        **kwargs,
    ) -> xr.DataArray | xr.Dataset | list[xr.DataArray]:
        loader, default_dir = self._get_current_defaults()

        if data_dir is None:
            data_dir = default_dir

        if not isinstance(identifier, int) and os.path.isfile(identifier):
            # If the identifier is a path to a file, ignore data_dir
            data_dir = None

        return loader.load(identifier, data_dir=data_dir, **kwargs)

    def summarize(
        self,
        data_dir: str | os.PathLike | None = None,
        usecache: bool = True,
        *,
        cache: bool = True,
        display: bool = True,
        **kwargs,
    ) -> xr.DataArray | xr.Dataset | list[xr.DataArray]:
        loader, default_dir = self._get_current_defaults()

        if data_dir is None:
            data_dir = default_dir

        return loader.summarize(
            data_dir, usecache, cache=cache, display=display, **kwargs
        )

    def _get_current_defaults(self):
        if self.current_loader is None:
            raise ValueError(
                "No loader has been set. Set a loader with `erlab.io.set_loader` first"
            )
        return self.current_loader, self.default_data_dir

    def __repr__(self) -> str:
        out = "Registered data loaders\n=======================\n\n"
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
        out = ""
        out += "<table><thead>"
        out += (
            "<tr>"
            "<th style='text-align:left;'><b>Name</b></th>"
            "<th style='text-align:left;'><b>Aliases</b></th>"
            "<th style='text-align:left;'><b>Loader class</b></th>"
            "</tr>"
        )
        out += "</thead><tbody>"
        for k, v in self.loaders.items():
            aliases = ", ".join(v.aliases) if v.aliases is not None else ""

            # May be either a class or an instance
            if isinstance(v, LoaderBase):
                v = type(v)

            cls_name = f"{v.__module__}.{v.__qualname__}"

            out += (
                "<tr>"
                f"<td style='text-align:left;'>{k}</td>"
                f"<td style='text-align:left;'>{aliases}</td>"
                f"<td style='text-align:left;'>{cls_name}</td>"
                "</tr>"
            )
        out += "</tbody></table>"

        return out

    load.__doc__ = LoaderBase.load.__doc__
    summarize.__doc__ = LoaderBase.summarize.__doc__
