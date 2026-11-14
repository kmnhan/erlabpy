"""Run Sphinx with a traceback before the Read the Docs build deadline."""

import faulthandler
import runpy

if __name__ == "__main__":
    faulthandler.dump_traceback_later(18 * 60)
    runpy.run_module("sphinx", run_name="__main__")
