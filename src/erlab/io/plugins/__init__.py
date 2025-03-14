"""Data loading plugins.

The modules in this package provide data loaders for various beamlines and laboratories.
Each module contains a class that subclasses :class:`erlab.io.dataloader.LoaderBase`,
which can be accessed through :attr:`erlab.io.loaders`.

See :doc:`/generated/erlab.io.dataloader` for more information on how to write a custom
loader.

.. currentmodule:: erlab.io.plugins

.. rubric:: Modules

.. autosummary::
   :toctree:

   da30
   erpes
   esm
   i05
   kriss
   lorea
   maestro
   mbs
   merlin
   ssrl52

"""

import importlib
import pathlib
import warnings


class PluginImportWarning(UserWarning):
    """Issued when a plugin fails to load."""


for path in pathlib.Path(__file__).resolve().parent.iterdir():
    if (
        path.is_file()
        and path.suffix == ".py"
        and not path.name.startswith((".", "__"))
    ):
        module_name = __name__ + "." + path.stem
        try:
            importlib.import_module(module_name)
        except Exception:
            warnings.warn(
                f"Failed to load '{module_name}'. "
                f"Import the module to trace the error.",
                PluginImportWarning,
                stacklevel=1,
            )
