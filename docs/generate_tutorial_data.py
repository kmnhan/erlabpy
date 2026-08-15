"""Generate deterministic data files used by the documentation tutorials."""

from __future__ import annotations

import pathlib

from erlab.io.exampledata import generate_data_angles


def main() -> None:
    output = (
        pathlib.Path(__file__).parent
        / "source"
        / "_static"
        / "tutorial-data"
        / "manager-quick-start.h5"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    data = generate_data_angles(
        shape=(112, 48, 128), assign_attributes=True, seed=1
    ).astype("float32")
    data = data.rename("example_map")
    data.attrs["description"] = (
        "Deterministic example data for the ImageTool Manager quick start"
    )
    data.to_netcdf(
        output,
        engine="h5netcdf",
        encoding={
            "example_map": {"compression": "gzip", "compression_opts": 4},
        },
    )


if __name__ == "__main__":
    main()
