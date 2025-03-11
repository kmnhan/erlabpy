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
    if "use_manager" in kwargs:
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
        manager = True

    if manager and not erlab.interactive.imagetool.manager.is_running():
        erlab.utils.misc.emit_user_level_warning(
            "The manager is not running. Opening the ImageTool window(s) directly."
        )

    if manager:
        erlab.interactive.imagetool.manager.show_in_manager(
            data, link=link, link_colors=link_colors, **kwargs
        )
        return None

    with erlab.interactive.utils.setup_qapp(execute) as execute:
        itool_list = [
            erlab.interactive.imagetool.ImageTool(d, **kwargs)
            for d in _parse_input(data)
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

    if execute:
        return None

    if len(itool_list) == 1:
        return itool_list[0]

    return itool_list
