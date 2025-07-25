__all__ = ["itool"]

from collections.abc import Collection

import numpy.typing as npt
import xarray as xr

import erlab
from erlab.interactive.imagetool.core import SlicerLinkProxy, _parse_input


def itool(
    data: Collection[xr.DataArray | npt.NDArray]
    | xr.DataArray
    | npt.NDArray
    | xr.Dataset
    | xr.DataTree,
    *,
    link: bool = False,
    link_colors: bool = True,
    manager: bool | None = None,
    replace: Collection[int] | int | None = None,
    execute: bool | None = None,
    **kwargs,
) -> (
    erlab.interactive.imagetool.ImageTool
    | list[erlab.interactive.imagetool.ImageTool]
    | None
):
    """Create and display ImageTool windows.

    This tool can also conveniently accessed with :meth:`xarray.DataArray.qshow` and
    :meth:`xarray.Dataset.qshow`.

    Parameters
    ----------
    data : DataArray, Dataset, DataTree, ndarray, list of DataArray or list of ndarray
        The data to be displayed. Data can be provided as:

        - A `xarray.DataArray` with 2 to 4 dimensions

          The DataArray will be displayed in an ImageTool window.

        - A numpy array with 2 to 4 dimensions

          The array will be converted to a DataArray and displayed in an ImageTool.

        - A list of the above objects

          Multiple ImageTool windows will be created and displayed.

        - A `xarray.Dataset`

          Every DataArray in the Dataset will be displayed across multiple ImageTool
          windows. Data variables that have less than 2 dimensions or more than 4
          dimensions are ignored. Dimensions with length 1 are automatically squeezed.

        - A `xarray.DataTree`

          Every leaf node will be parsed as a `xarray.Dataset`.
    link
        Whether to enable linking between multiple ImageTool windows when `data` is a
        sequence or a `xarray.Dataset`, by default `False`.
    link_colors
        Whether to link the color maps between multiple linked ImageTool windows, by
        default `True`. This argument has no effect if `link` is set to `False`.
    manager
        Whether to open the ImageTool window(s) using the :class:`ImageToolManager
        <erlab.interactive.imagetool.manager.ImageToolManager>` if it is running. If not
        provided, the manager will only be used if it is in the same process as the
        caller.

        .. versionchanged:: 3.4.0

            Argument renamed from ``use_manager`` to ``manager``.
    replace
        When using the manager, this argument specifies which existing ImageTool windows
        should be replaced with the new data. If the manager is not used, this argument
        is ignored. ``replace`` can be set to:

        - `None` (default):
            No existing windows are replaced. New windows are created for the new data.

        - A single integer:
            - A valid index of an existing ImageTool window:
                The data in the window with the specified index is replaced with the new
                data.

            - A number that is greater by 1 than the largest existing index:
                A new ImageTool window is created with the new data. This is useful when
                you want to add a new window on initial execution, but want to replace
                the window with the same index on subsequent calls.

            - A negative integer:
                The index is interpreted as an index from the end of the list of
                existing ImageTool windows, sorted by their indices. For example, ``-1``
                refers to the window with the largest index, ``-2`` to the second
                largest, and so on.

        - A list of integers:
            A list of integers specifying the indices of the windows to be replaced,
            each of which is interpreted as described above. The length of the list must
            match the number of windows ``data`` is expected to create.

        If this argument is used, the ``link``, ``link_colors``, and ``kwargs``
        arguments are ignored, since no new windows are created.
    execute
        Whether to execute the Qt event loop and display the window, by default `None`.
        For more information, see :func:`erlab.interactive.utils.setup_qapp`.

        This argument has no effect when the ImageTool window(s) are opened in the
        manager.
    **kwargs
        Additional keyword arguments to be passed onto the underlying slicer area. For a
        full list of supported arguments, see the
        `erlab.interactive.imagetool.core.ImageSlicerArea` documentation.

    Returns
    -------
    ImageTool or list of ImageTool or None
        The created ImageTool window(s).

        If the window(s) are executed, the function will return `None`, since the event
        loop will prevent the function from returning until the window(s) are closed.

        If the window(s) are not executed, for example while running in an IPython shell
        with ``%gui qt``, the function will not block and return the ImageTool window(s)
        or a list of ImageTool windows depending on the input data.

        The function will also return `None` if the windows are opened in the
        :class:`ImageToolManager
        <erlab.interactive.imagetool.manager.ImageToolManager>`.

    Examples
    --------
    >>> itool(data, cmap="gray", gamma=0.5)
    >>> itool([data1, data2], link=True)
    """
    if "use_manager" in kwargs:  # pragma: no cover
        # Deprecated argument, remove in future releases
        manager = kwargs.pop("use_manager")
        erlab.utils.misc.emit_user_level_warning(
            "The `use_manager` argument has been renamed to `manager`."
            "Support for the old argument will be removed in a future release.",
            category=FutureWarning,
        )

    if (
        manager is None
        and erlab.interactive.imagetool.manager.is_running()
        and erlab.interactive.imagetool.manager._manager_instance is not None
    ):
        # Called from the same process as the manager, using the manager by default
        manager = True

    if manager and not erlab.interactive.imagetool.manager.is_running():
        erlab.utils.misc.emit_user_level_warning(
            "The manager is not running. Opening the ImageTool window(s) directly."
        )

    if manager:
        if replace is not None:
            erlab.interactive.imagetool.manager.replace_data(index=replace, data=data)
        else:
            erlab.interactive.imagetool.manager.show_in_manager(
                data, link=link, link_colors=link_colors, **kwargs
            )
        return None

    data_parsed = _parse_input(data)
    with erlab.interactive.utils.setup_qapp(execute) as execute:
        itool_list = [
            erlab.interactive.imagetool.ImageTool(d, **kwargs) for d in data_parsed
        ]

        for w in itool_list:
            w.show()

        if link:
            linker = SlicerLinkProxy(  # noqa: F841
                *[w.slicer_area for w in itool_list], link_colors=link_colors
            )
            # TODO: make sure this is not garbage collected

        itool_list[-1].activateWindow()
        itool_list[-1].raise_()

    if execute:  # pragma: no cover
        return None

    if len(itool_list) == 1:
        return itool_list[0]

    return itool_list
