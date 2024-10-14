r"""Base functionality for implementing data loaders.

This module provides a base class :class:`LoaderBase` for implementing data loaders.
Data loaders are plugins used to load data from various file formats.

Each data loader is a subclass of :class:`LoaderBase` that must implement several
methods and attributes.

A detailed guide on how to implement a data loader can be found in the :ref:`User Guide
<user-guide/io:Implementing a data loader plugin>`.
"""

from __future__ import annotations

import contextlib
import errno
import importlib
import itertools
import os
import pathlib
import warnings
from collections.abc import Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Self,
    TypeGuard,
    TypeVar,
    cast,
    overload,
)

import numpy as np
import pandas
import xarray as xr
from xarray.core.datatree import DataTree

from erlab.utils.formatting import format_html_table, format_value

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        ItemsView,
        Iterable,
        Iterator,
        KeysView,
        Mapping,
    )


_T = TypeVar("_T")


def _is_sequence_of(val: Any, element_type: type[_T]) -> TypeGuard[Sequence[_T]]:
    return all(isinstance(x, element_type) for x in val) and isinstance(val, Sequence)


class ValidationWarning(UserWarning):
    """Issued when the loaded data fails validation checks."""


class ValidationError(Exception):
    """Raised when the loaded data fails validation checks."""


class LoaderNotFoundError(Exception):
    """Raised when a loader is not found in the registry."""

    def __init__(self, key: str) -> None:
        super().__init__(f"Loader for name or alias {key} not found in the registry")


class _Loader(type):
    """Metaclass for data loaders.

    This metaclass wraps the `identify` method to raise an informative
    `FileNotFoundError` when subclasses of DataLoaderBase return None or an empty list.
    """

    def __new__(cls, name, bases, dct):
        # Create class
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
                return result

            new_class.identify = wrapped_identify

        return new_class


class LoaderBase(metaclass=_Loader):
    """Base class for all data loaders."""

    name: str
    """
    Name of the loader. Using a unique and descriptive name is recommended. For easy
    access, it is recommended to use a name that passes :meth:`str.isidentifier`.

    Notes
    -----
    - Changing the name of a loader is not recommended as it may break existing code. If
      a different name is required, Add an alias instead.
    - Loaders with the name prefixed with an underscore are not registered.
    """

    aliases: Iterable[str] | None = None
    """Alternative names for the loader."""

    name_map: ClassVar[dict[str, str | Iterable[str]]] = {}
    """
    Dictionary that maps **new** coordinate or attribute names to **original**
    coordinate or attribute names. If there are multiple possible names for a single
    attribute, the value can be passed as an iterable.

    Note
    ----
    - Non-dimension coordinates in the resulting data will try to follow the order of
      the keys in this dictionary.
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

    additional_attrs: ClassVar[dict[str, str | int | float]] = {}
    """Additional attributes to be added to the data after loading.

    Notes
    -----
    - The attributes are added after renaming with :meth:`process_keys
      <erlab.io.dataloader.LoaderBase.process_keys>`, so keys will appear in the data as
      provided.
    - If an attribute with the same name is already present in the data, it is skipped.
    """

    additional_coords: ClassVar[dict[str, str | int | float]] = {}
    """Additional non-dimension coordinates to be added to the data after loading.

    Notes
    -----
    - The coordinates are added after renaming with :meth:`process_keys
      <erlab.io.dataloader.LoaderBase.process_keys>`, so keys will appear in the data as
      provided.
    - If a coordinate with the same name is already present in the data, it is skipped.
    """

    formatters: ClassVar[dict[str, Callable]] = {}
    """Mapping from attribute names (after renaming) to custom formatters.

    The formatters must take the attribute value and return a value that can be
    converted to a string with :meth:`value_to_string
    <erlab.io.dataloader.LoaderBase.value_to_string>`. The resulting formats are used
    for human readable display of some attributes in the summary table and the
    information accessor.

    Note
    ----
    The formatters are only used for display purposes and do not affect the stored data.
    """

    always_single: bool = True
    """
    Setting this to `True` disables implicit loading of multiple files for a single
    scan. This is useful for setups where each scan is always stored in a single file.
    """

    skip_validate: bool = False
    """
    If `True`, validation checks will be skipped. If `False`, data will be checked with
    :meth:`validate <erlab.io.dataloader.LoaderBase.validate>` every time it is loaded.
    """

    strict_validation: bool = False
    """
    If `True`, validation checks will raise a `ValidationError` on the first failure
    instead of warning. Useful for debugging data loaders. This has no effect if
    `skip_validate` is `True`.
    """

    @property
    def name_map_reversed(self) -> dict[str, str]:
        """Reverse of :attr:`name_map <erlab.io.dataloader.LoaderBase.name_map>`.

        This property is useful for mapping original names to new names.
        """
        return self._reverse_mapping(self.name_map)

    @property
    def file_dialog_methods(self) -> dict[str, tuple[Callable, dict[str, Any]]]:
        """Map from file dialog names to the loader method and its arguments.

        Subclasses can override this property to provide support for loading data from
        the load menu of the ImageTool GUI.

        Returns
        -------
        loader_mapping
            A dictionary mapping the file dialog names to a tuple of length 2 containing
            the data loading function and arguments.

            The first item of the tuple should be a callable that takes the first
            positional argument as a path to a file, usually ``self.load``.

            The second item should be a dictionary containing keyword arguments to be
            passed to the method.

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

        The default behavior formats the given value with :func:`format_value
        <erlab.utils.formatting.format_value>`. Override this classmethod to change the
        printed format of each cell.

        """
        return format_value(val)

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
            Summary dataframe as returned by `generate_summary`.

        Returns
        -------
        pandas.io.formats.style.Styler
            The styler to be displayed.

        """
        style = df.style.format(cls.value_to_string)

        hidden = [c for c in ("Time", "Path") if c in df.columns]
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
        **kwargs,
    ) -> (
        xr.DataArray
        | xr.Dataset
        | DataTree
        | list[xr.DataArray]
        | list[xr.Dataset]
        | list[DataTree]
    ):
        """Load ARPES data.

        Parameters
        ----------
        identifier
            Value that identifies a scan uniquely.

            - If a string or path-like object is given, it is assumed to be the path to
              the data file relative to `data_dir`. If `data_dir` is not specified, it
              is assumed to be the full path to the data file.

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
            `xarray.Dataset`, or a `DataTree`.

            This argument is only used when `single` is `False`.
        parallel
            Whether to load multiple files in parallel. If not specified, files are
            loaded in parallel only when there are more than 15 files to load.
        **kwargs
            Additional keyword arguments are passed to `identify`.

        Returns
        -------
        `xarray.DataArray` or `xarray.Dataset` or `DataTree`
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

            import erlab.io

            erlab.io.set_data_dir("data")
            erlab.io.load("example.txt")

          However, if ``./data/example.txt`` also exists, the same code will load that
          one instead while warning about the ambiguity. This behavior may lead to
          unexpected results when the directory structure is not organized. Keep this in
          mind and try to keep all data files in the same level.

        """
        if self.always_single:
            single = True

        if isinstance(identifier, int):
            if data_dir is None:
                raise ValueError(
                    "data_dir must be specified when identifier is an integer"
                )
            file_paths, coord_dict = cast(
                tuple[list[str], dict[str, Sequence]],
                self.identify(identifier, data_dir, **kwargs),
            )  # Return type enforced by metaclass, cast to avoid mypy error
            # file_paths: list of file paths with at least one element

            if len(file_paths) == 1:
                # Single file resolved
                data: xr.DataArray | xr.Dataset | DataTree = self.load_single(
                    file_paths[0]
                )
            else:
                # Multiple files resolved
                if combine:
                    data = self.combine_multiple(
                        self.load_multiple_parallel(file_paths, parallel=parallel),
                        coord_dict,
                    )
                else:
                    return self.load_multiple_parallel(
                        file_paths, parallel=parallel, post_process=True
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
                    return self.load(
                        new_identifier, new_dir, single=single, **new_kwargs
                    )
                # On failure, assume single file
                single = True

            data = self.load_single(identifier)

        data = self.post_process_general(data)

        if not self.skip_validate:
            self.validate(data)

        return data

    def get_formatted_attr_or_coord(
        self, data: xr.DataArray, attr_or_coord_name: str
    ) -> Any:
        """Return the formatted value of the given attribute or coordinate.

        The value is formatted using the function specified in :attr:`formatters
        <erlab.io.dataloader.LoaderBase.formatters>`. If the name is not found, an empty
        string is returned.

        Parameters
        ----------
        data : DataArray
            The data to extract the attribute or coordinate from.
        attr_or_coord_name : str
            The name of the attribute or coordinate to extract.

        """
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
            Additional keyword arguments to be passed to :meth:`generate_summary
            <erlab.io.dataloader.LoaderBase.generate_summary>`.

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
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), str(data_dir)
            )

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

    def isummarize(self, df: pandas.DataFrame | None = None, **kwargs) -> None:
        """Display an interactive summary.

        This method provides an interactive summary of the data using ipywidgets and
        matplotlib.

        Parameters
        ----------
        df
            A summary dataframe as returned by :meth:`generate_summary
            <erlab.io.dataloader.LoaderBase.generate_summary>`. If `None`, a dataframe
            will be generated using :meth:`summarize
            <erlab.io.dataloader.LoaderBase.summarize>`. Defaults to `None`.
        **kwargs
            Additional keyword arguments to be passed to :meth:`summarize
            <erlab.io.dataloader.LoaderBase.summarize>` if `df` is None.

        Note
        ----
        This method requires `ipywidgets` to be installed.

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

        # Temporary variable to store loaded data
        self._temp_data: xr.DataArray | None = None
        # !TODO: properly GC this variable

        def _format_data_info(series: pandas.Series) -> str:
            # Format data info as HTML table
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
                table += f"<td style='text-align:left;'>{self.value_to_string(v)}</td>"
                table += "</tr>"

            table += "</tbody></table>"
            table += "</div>"
            return table

        def _update_data(_, *, full: bool = False) -> None:
            # Load data for selected row
            series = df.loc[data_select.value]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                _path = series["Path"]
                _basename = os.path.basename(_path)
                _dirname = os.path.dirname(_path)

                full_button.disabled = True

                if not self.always_single:
                    idx, _ = self.infer_index(os.path.splitext(_basename)[0])
                    if idx is not None:
                        ident = self.identify(idx, _dirname)
                        if ident is not None:
                            n_scans = len(ident[0])
                            if n_scans > 1 and not full:
                                full_button.disabled = False

                out = self.load(_basename, _dirname, single=not full)
                if isinstance(out, xr.DataArray):
                    self._temp_data = out
                del out

                data_info.value = _format_data_info(series)
            if self._temp_data is None:
                return

            if self._temp_data.ndim == 4:
                # If the data is 4D, average over the last dimension, making it 3D
                self._temp_data = self._temp_data.mean(str(self._temp_data.dims[-1]))
                # !TODO: Add 2 sliders for 4D data

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

            with out:
                plot_data.qplot(ax=plt.gca())
                plt.title("")  # Remove automatically generated title

                # Add line at Fermi level if the data is 2D and has an energy dimension
                # that includes zero
                if (plot_data.ndim == 2 and "eV" in plot_data.dims) and (
                    plot_data["eV"].values[0] * plot_data["eV"].values[-1] < 0
                ):
                    eplt.fermiline(
                        orientation="h" if plot_data.dims[0] == "eV" else "v"
                    )
                show_inline_matplotlib_plots()

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
        prev_button.on_click(_prev)
        next_button.on_click(_next)
        full_button.on_click(lambda _: _update_data(None, full=True))
        if self.always_single:
            buttons = [prev_button, next_button]
        else:
            buttons = [prev_button, next_button, full_button]

        # List of data files
        data_select = Select(options=list(df.index), value=next(iter(df.index)), rows=8)
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
        self, file_path: str | os.PathLike
    ) -> xr.DataArray | xr.Dataset | DataTree:
        r"""Load a single file and return it as an xarray data structure.

        Any scan-specific postprocessing should be implemented in this method.

        This method must be implemented to return the *smallest possible data structure*
        that represents the data in a single file. For instance, if a single file
        contains a single scan region, the method should return a single
        `xarray.DataArray`. If it contains multiple regions, the method should return a
        `xarray.Dataset` or `DataTree` depending on whether the regions can be merged
        with without conflicts (i.e., all mutual coordinates of the regions are the
        same).

        Parameters
        ----------
        file_path
            Full path to the file to be loaded.

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
          <erlab.io.dataloader.LoaderBase.identify>` so that they can be combined with
          :meth:`combine_multiple <erlab.io.dataloader.LoaderBase.combine_multiple>`.
          This should not be a problem since in most cases, the data structure of
          associated files acquired during the same scan will be identical.
        - For `DataTree` objects, returned trees must be named with a unique identifier
          to avoid conflicts when combining.
        """
        raise NotImplementedError("method must be implemented in the subclass")

    def identify(
        self, num: int, data_dir: str | os.PathLike
    ) -> tuple[list[str], dict[str, Sequence]] | None:
        """Identify the files and coordinates for a given scan number.

        This method takes a scan index and transforms it into a list of file paths and
        coordinates. For scans spread over multiple files, the coordinates must be a
        dictionary mapping scan axes names to scan coordinates. For single file scans,
        the list should contain only one file path and coordinates must be an empty
        dictionary.

        The keys of the coordinates must match the coordinate name conventions used by
        the data returned by :meth:`load_single
        <erlab.io.dataloader.LoaderBase.load_single>`. For example, if
        :meth:`load_single <erlab.io.dataloader.LoaderBase.load_single>` is implemented
        so that it renames properties and coordinates using :meth:`process_keys
        <erlab.io.dataloader.LoaderBase.process_keys>`, the dictionary must also be
        transformed to new names prior to returning by using the mapping returned by
        :attr:`name_map_reversed <erlab.io.dataloader.LoaderBase.name_map_reversed>`.

        If no files are found for the given parameters, `None` should be returned.

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
        coord_dict : dict[str, Sequence]
            A dictionary mapping scan axes names to scan coordinates. For scans spread
            over multiple files, the coordinates will be sequences, with each element
            corresponding to each file in ``files``. For single file scans, an empty
            dictionary must be returned.

        """
        raise NotImplementedError("method must be implemented in the subclass")

    def infer_index(self, name: str) -> tuple[int | None, dict[str, Any]]:
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
            Additional keyword arguments to be passed to :meth:`load
            <erlab.io.dataloader.LoaderBase.load>` when the index is found. This
            argument is useful when the index alone is not enough to load the data.

        Note
        ----
        For loaders with :attr:`always_single
        <erlab.io.dataloader.LoaderBase.always_single>` set to `True`, this method is
        not used.

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
        raise NotImplementedError(
            f"loader '{self.name}' does not support folder summaries"
        )

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

    @overload
    def combine_multiple(
        self,
        data_list: list[xr.DataArray],
        coord_dict: dict[str, Sequence],
    ) -> xr.DataArray: ...

    @overload
    def combine_multiple(
        self,
        data_list: list[xr.Dataset],
        coord_dict: dict[str, Sequence],
    ) -> xr.Dataset: ...

    @overload
    def combine_multiple(
        self,
        data_list: list[DataTree],
        coord_dict: dict[str, Sequence],
    ) -> DataTree: ...

    def combine_multiple(
        self,
        data_list: list[xr.DataArray] | list[xr.Dataset] | list[DataTree],
        coord_dict: dict[str, Sequence],
    ) -> xr.DataArray | xr.Dataset | DataTree:
        if _is_sequence_of(data_list, DataTree):
            raise NotImplementedError(
                "Combining DataTrees into a single tree "
                "will be supported in a future release"
            )

        if len(coord_dict) == 0:
            # No coordinates to combine given
            # Multiregion scans over multiple files may be provided like this

            if _is_sequence_of(data_list, DataTree):
                pass
            else:
                try:
                    return xr.combine_by_coords(
                        cast(Sequence[xr.DataArray] | Sequence[xr.Dataset], data_list),
                        combine_attrs=self.combine_attrs,
                        join="exact",
                    )
                except Exception as e:
                    raise RuntimeError(
                        "Failed to combine data. Try passing "
                        "`combine=False` to `erlab.io.load`"
                    ) from e

        if _is_sequence_of(data_list, xr.DataArray) or _is_sequence_of(
            data_list, xr.Dataset
        ):
            # Rename with process_keys after assigning coords and expanding dims
            # This is necessary to ensure that coordinate_attrs are preserved
            combined = xr.combine_by_coords(
                [
                    self.process_keys(
                        data.assign_coords(
                            {k: v[i] for k, v in coord_dict.items()}
                        ).expand_dims(tuple(coord_dict.keys()))
                    )
                    for i, data in enumerate(data_list)
                ],
                combine_attrs=self.combine_attrs,
            )
            if (
                isinstance(combined, xr.Dataset)
                and len(combined.data_vars) == 1
                and combined.attrs == {}
            ):
                # Named DataArrays combined into a Dataset, extract the DataArray
                var_name = next(iter(combined.data_vars))
                combined = combined[var_name]

                if combined.name is None:
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
            key_mapping = self.name_map_reversed

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

        """
        darr = self.process_keys(darr)

        for k in self.average_attrs:
            if k in darr.coords:
                v = darr[k].values.mean()
                darr = darr.drop_vars(k).assign_attrs({k: v})

        new_attrs = {
            k: v for k, v in self.additional_attrs.items() if k not in darr.attrs
        }
        new_attrs["data_loader_name"] = str(self.name)
        darr = darr.assign_attrs(new_attrs)

        new_coords = {
            k: v for k, v in self.additional_coords.items() if k not in darr.coords
        }
        return darr.assign_coords(new_coords)

    def _reorder_coords(self, darr: xr.DataArray):
        """Sort the coordinates of the given DataArray."""
        ordered_coords = {}
        coord_dict = dict(darr.coords)
        for d in darr.dims:
            # Move dimension coords to the front
            ordered_coords[d] = coord_dict.pop(d)

        for d in itertools.chain(self.name_map.keys(), self.additional_coords.keys()):
            if d in coord_dict:
                ordered_coords[d] = coord_dict.pop(d)
        ordered_coords = ordered_coords | coord_dict

        return xr.DataArray(
            darr.values, coords=ordered_coords, dims=darr.dims, attrs=darr.attrs
        )

    @overload
    def post_process_general(self, data: xr.DataArray) -> xr.DataArray: ...

    @overload
    def post_process_general(self, data: xr.Dataset) -> xr.Dataset: ...

    @overload
    def post_process_general(self, data: DataTree) -> DataTree: ...

    def post_process_general(
        self, data: xr.DataArray | xr.Dataset | DataTree
    ) -> xr.DataArray | xr.Dataset | DataTree:
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
            - If a `DataTree`, the post-processing is applied to each leaf node
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

        if isinstance(data, DataTree):
            return cast(DataTree, data.map_over_subtree(self.post_process_general))

        raise TypeError(
            "data must be a DataArray, Dataset, or DataTree, but got " + type(data)
        )

    @classmethod
    def validate(cls, data: xr.DataArray | xr.Dataset | DataTree) -> None:
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
            The data to be validated. If a `Dataset` or `DataTree` is passed, validation
            is performed on each data variable recursively.

        """
        if isinstance(data, xr.Dataset):
            for v in data.data_vars.values():
                cls.validate(v)
            return

        if isinstance(data, DataTree):
            data.map_over_subtree(cls.validate)
            return

        for c in ("beta", "delta", "xi", "hv"):
            if c not in data.coords:
                cls._raise_or_warn(f"Missing coordinate {c}")

        for a in ("configuration", "temp_sample"):
            if a not in data.attrs:
                cls._raise_or_warn(f"Missing attribute {a}")

        if "configuration" not in data.attrs:
            return

        if data.attrs["configuration"] not in (1, 2):
            if data.attrs["configuration"] not in (3, 4):
                cls._raise_or_warn(
                    f"Invalid configuration {data.attrs['configuration']}"
                )
            elif "chi" not in data.coords:
                cls._raise_or_warn("Missing coordinate chi")

    def load_multiple_parallel(
        self,
        file_paths: list[str],
        parallel: bool | None = None,
        post_process: bool = False,
    ) -> list[xr.DataArray] | list[xr.Dataset] | list[DataTree]:
        """Load multiple files in parallel.

        Parameters
        ----------
        file_paths
            A list of file paths to load.
        parallel
            If `True`, data loading will be performed in parallel using `dask.delayed`.
        post_process
            Whether to post-process each data object after loading.

        Returns
        -------
        A list of the loaded data.
        """
        if parallel is None:
            parallel = len(file_paths) > 15

        if post_process:

            def _load_func(filename):
                return self.load(filename, single=True)

        else:
            _load_func = self.load_single

        if parallel:
            import joblib

            return joblib.Parallel(n_jobs=-1, max_nbytes=None)(
                joblib.delayed(_load_func)(f) for f in file_paths
            )

        return [_load_func(f) for f in file_paths]

    @classmethod
    def _raise_or_warn(cls, msg: str) -> None:
        if cls.strict_validation:
            raise ValidationError(msg)
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

    default_data_dir: pathlib.Path | None = None
    """Default directory to search for data files \n\n:meta hide-value:"""

    def _register(self, loader_class: type[LoaderBase]) -> None:
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

    def __iter__(self) -> Iterator[str]:
        return iter(self.loaders)

    def __getitem__(self, key: str) -> LoaderBase:
        return self.get(key)

    def __getattr__(self, key: str) -> LoaderBase:
        try:
            return self.get(key)
        except LoaderNotFoundError as e:
            raise AttributeError(str(e)) from e

    def set_loader(self, loader: str | LoaderBase | None) -> None:
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

    def set_data_dir(self, data_dir: str | os.PathLike | None) -> None:
        """Set the default data directory for the data loader.

        All subsequent calls to :func:`erlab.io.load` will use the `data_dir` set here
        unless specified.

        Parameters
        ----------
        data_dir
            The path to a directory.

        Note
        ----
        This will only affect :func:`erlab.io.load`. If the loader's ``load`` method is
        called directly, it will not use the default data directory.

        """
        if data_dir is None:
            self.default_data_dir = None
            return

        self.default_data_dir = pathlib.Path(data_dir).resolve(strict=True)

    def load(
        self,
        identifier: str | os.PathLike | int,
        data_dir: str | os.PathLike | None = None,
        **kwargs,
    ) -> (
        xr.DataArray
        | xr.Dataset
        | DataTree
        | list[xr.DataArray]
        | list[xr.Dataset]
        | list[DataTree]
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
                warnings.warn(
                    f"Found {identifier!s} in the default directory "
                    f"{default_dir!s}, but conflicting file {abs_file!s} was found. "
                    "The first file will be loaded. "
                    "Consider specifying the directory explicitly.",
                    stacklevel=2,
                )
            else:
                # If the identifier is a path to a file, ignore default_dir
                default_dir = None

        if data_dir is None:
            data_dir = default_dir

        return loader.load(identifier, data_dir=data_dir, **kwargs)

    def summarize(
        self,
        data_dir: str | os.PathLike | None = None,
        usecache: bool = True,
        *,
        cache: bool = True,
        display: bool = True,
        **kwargs,
    ) -> pandas.DataFrame | pandas.io.formats.style.Styler | None:
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
                f"{k}: {tuple(v.aliases)}"
                for k, v in self.loaders.items()
                if v.aliases is not None
            ]
        )
        return out

    def _repr_html_(self) -> str:
        rows: list[tuple[str, str, str]] = [("Name", "Aliases", "Loader class")]

        for k, v in self.loaders.items():
            aliases = ", ".join(v.aliases) if v.aliases is not None else ""

            # May be either a class or an instance
            if isinstance(v, LoaderBase):
                v = type(v)

            cls_name = f"{v.__module__}.{v.__qualname__}"
            rows.append((k, aliases, cls_name))

        return format_html_table(rows, header_rows=1)

    load.__doc__ = LoaderBase.load.__doc__
    summarize.__doc__ = LoaderBase.summarize.__doc__
