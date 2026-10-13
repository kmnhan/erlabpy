"""Data loader for the ESPRESSO endstation at HiSOR BL9B."""

__all__ = ["ESPRESSOLoader"]

from erlab.io.plugins.da30 import DA30Loader


class ESPRESSOLoader(DA30Loader):
    """Load Scienta analyzer data from the ESPRESSO endstation at HiSOR BL9B."""

    name = "espresso"
    description = "HiSOR BL9B ESPRESSO"
    coordinate_attrs = ("beta", "hv")

    @property
    def file_dialog_methods(self):
        return {"ESPRESSO Data (*.ibw *.pxt *.txt *.zip)": (self.load, {})}
