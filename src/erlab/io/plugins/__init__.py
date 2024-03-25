"""Data loading plugins.

The modules in this package provide data loaders for various beamlines and laboratories.
Each module contains a class that subclasses :class:`erlab.io.dataloader.LoaderBase`,
which can be accessed through :attr:`erlab.io.loaders`.

See :doc:`/generated/erlab.io.dataloader` for more information on how to write a custom
loader.

.. currentmodule:: erlab.io.plugins

Modules
=======

.. autosummary::
   :toctree:

   merlin
   ssrl52
   da30
   kriss

"""

import importlib
import os
import traceback

for fname in os.listdir(os.path.dirname(os.path.abspath(__file__))):
    if (
        not fname.startswith(".")
        and not fname.startswith("__")
        and fname.endswith(".py")
    ):
        try:
            importlib.import_module(__name__ + "." + os.path.splitext(fname)[0])
        except Exception as e:
            print(e)
            traceback.print_exc()
