"""Quickly browse and load ARPES data files with a file manager-like interface.

.. image:: ../images/explorer_light.png
    :align: center
    :alt: Data explorer window in light mode
    :class: only-light

.. only:: format_html

    .. image:: ../images/explorer_dark.png
        :align: center
        :alt: Data explorer window in dark mode
        :class: only-dark
"""

import os

import erlab
from erlab.interactive.explorer._tabbed_explorer import _TabbedExplorer


def data_explorer(
    directory: str | os.PathLike | None = None,
    loader_name: str | None = None,
    *,
    execute: bool | None = None,
) -> None:
    """Start the data explorer.

    Data explorer is a tool to browse and load ARPES data files with a file manager-like
    interface. Data attributes of supported files can be quickly inspected, and can be
    loaded into ImageTool manager for further analysis.

    The data explorer can be started from the command line as a standalone application
    with the following command:

    .. code-block:: bash

        python -m erlab.interactive.explorer

    Also, it can be opened from the GUI by selecting "File" -> "Data Explorer" or by
    pressing :kbd:`Ctrl+E` in ImageTool manager.

    Parameters
    ----------
    directory
        Initial directory to display in the explorer.
    loader_name
        Name of the loader to use to load the data. The loader must be registered in
        :attr:`erlab.io.loaders`.
    """
    with erlab.interactive.utils.setup_qapp(execute):
        win = _TabbedExplorer(root_path=directory, loader_name=loader_name)
        win.show()
        win.raise_()
        win.activateWindow()
