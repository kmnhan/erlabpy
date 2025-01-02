"""Interactive data browser.

.. currentmodule:: erlab.interactive.imagetool

Modules
=======

.. autosummary::
   :toctree:

   mainwindow
   core
   slicer
   fastbinning
   controls
   dialogs
   manager

"""

from collections.abc import Collection
from typing import TYPE_CHECKING

import lazy_loader as _lazy
import numpy.typing as npt
import xarray as xr

import erlab
from erlab.interactive.imagetool.core import (
    SlicerLinkProxy,
    _parse_input,
)
from erlab.utils.misc import emit_user_level_warning

__getattr__, __dir_lazy__, __all_lazy__ = _lazy.attach(
    __name__,
    submodules=["manager"],
    submod_attrs={
        "mainwindow": ["BaseImageTool", "ImageTool"],
    },
)

__all__ = [*__all_lazy__, "itool"]


def __dir__() -> list[str]:
    return [*__dir_lazy__(), "itool"]


if TYPE_CHECKING:
    from erlab.interactive.imagetool import manager  # noqa: F401
    from erlab.interactive.imagetool.mainwindow import (  # noqa: F401
        BaseImageTool,
        ImageTool,
    )


def itool(
    data: Collection[xr.DataArray | npt.NDArray]
    | xr.DataArray
    | npt.NDArray
    | xr.Dataset
    | xr.DataTree,
    *,
    link: bool = False,
    link_colors: bool = True,
    use_manager: bool = False,
    execute: bool | None = None,
    **kwargs,
) -> (
    erlab.interactive.imagetool.ImageTool
    | list[erlab.interactive.imagetool.ImageTool]
    | None
):
    """Create and display ImageTool windows.

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
    use_manager
        Whether to open the ImageTool window(s) using the :class:`ImageToolManager
        <erlab.interactive.imagetool.manager.ImageToolManager>` if it is running.
    execute
        Whether to execute the Qt event loop and display the window, by default `None`.
        If `None`, the execution is determined based on the current IPython shell. This
        argument has no effect if the :class:`ImageToolManager
        <erlab.interactive.imagetool.manager.ImageToolManager>` is running and
        `use_manager` is set to `True`. In most cases, the default value should be used.
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
    manager_running: bool = erlab.interactive.imagetool.manager.is_running()
    if (
        manager_running
        and erlab.interactive.imagetool.manager._manager_instance is not None
    ):
        use_manager = True

    if use_manager and not manager_running:
        use_manager = False
        emit_user_level_warning(
            "The manager is not running. Opening the ImageTool window(s) directly."
        )

    if use_manager:
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
