"""Interactive tools based on Qt and pyqtgraph.

.. currentmodule:: erlab.interactive

This module provides interactive tools for plotting and analyzing ARPES data. See the
sidebar on the right for a list of available tools.

Commonly used tools are available directly in the ``erlab.interactive`` namespace, so
regular users should not need to import the submodules directly.

Documentation of classes and functions in submodules mostly contain implementation
details for advanced users who want to create new interactive tools. A user guide for
creating new interactive tools will be available in the future. In the meantime, take a
look at the source code of :mod:`erlab.interactive.utils` and
:mod:`erlab.interactive.colors` which provide general utility functions for creating new
interactive tools.

.. rubric:: Modules

.. autosummary::
   :toctree: generated

   imagetool
   bzplot
   colors
   curvefittingtool
   derivative
   explorer
   fermiedge
   kspace
   utils

"""

try:
    import qtpy
except ImportError as e:
    raise ImportError(
        "A Qt binding is required for interactive tools. "
        "Please install one of the following packages:\n"
        "  - PySide6: 'pip install PySide6'\n"
        "  - PyQt6: 'pip install PyQt6'\n"
        "For more information, visit the official documentation of these packages."
    ) from e
else:
    if qtpy.QT5:
        raise ImportError(
            f"{qtpy.API_NAME} is no longer supported by erlabpy. "
            f"Please install PySide6 or PyQt6 and uninstall {qtpy.API_NAME}."
        )

import lazy_loader as _lazy

__getattr__, __dir__, __all__ = _lazy.attach_stub(__name__, __file__)


def load_ipython_extension(ipython) -> None:
    # %itool magic
    from erlab.interactive.imagetool._magic import ImageToolMagics

    ipython.register_magics(ImageToolMagics)

    # %watch magic
    from erlab.interactive.imagetool.manager._watcher import WatcherMagics

    watcher_magics = WatcherMagics(ipython)
    ipython.register_magics(watcher_magics)
    ipython.events.register("post_run_cell", watcher_magics._watcher._maybe_push)

    # Other tools
    from erlab.interactive._magic import InteractiveToolMagics

    ipython.register_magics(InteractiveToolMagics)


def unload_ipython_extension(ipython) -> None:
    watcher_magics = ipython.magics_manager.registry.get("WatcherMagics")
    watcher_magics._watcher.stop_watching_all()
    watcher_magics._watcher.shutdown()
    ipython.events.unregister("post_run_cell", watcher_magics._watcher._maybe_push)
