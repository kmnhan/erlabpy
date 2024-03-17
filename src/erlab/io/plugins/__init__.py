"""Data loading plugins.

.. currentmodule:: erlab.io.plugins

Modules
=======

.. autosummary::
   :toctree:
   
   merlin
   ssrl52

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
        except Exception:
            traceback.print_exc()
