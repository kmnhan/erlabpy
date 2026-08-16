# Data inspection and selection

Use these guides when you must prepare a cut or spectrum from measured coordinate
ranges, follow a path through momentum space, or compare several slices.

(how-to-python-select-average-data)=

## Averaging a cut or spectrum over coordinate ranges

Prepare an energy-momentum cut by cropping its displayed coordinates and averaging over
a finite `ky` window:

```python
import erlab

cut = data.qsel(
    kx=slice(-0.3, 0.3),
    ky=0.3,
    ky_width=0.06,
    eV=slice(-0.25, 0.05),
)
```

To prepare a local EDC around one point in the momentum plane, average all points inside
a radius stated in the same momentum units:

```python
edc = data.qsel.around(0.06, kx=0.52, ky=0.3)
```

```{eval-rst}
.. plot:: how_to/inspection_and_selection.py compare_radial_neighborhoods
   :include-source: false
   :alt: Constant energy surface with four circular averaging regions beside the resulting energy distribution curves
```

If the energy range is already selected, use
{meth}`qsel.mean <xarray.DataArray.qsel.mean>` to keep the mean energy as a scalar
coordinate:

```python
energy_window = data.sel(eV=slice(-0.025, 0.025))
constant_energy_map = energy_window.qsel.mean("eV")
```

Check the selected coordinate bounds and the number of averaged points before using the
result. Use the {doc}`Python workflow tutorial <../../tutorials/python/index>` for the
basic {meth}`qsel <xarray.DataArray.qsel>` sequence. A coordinate slice crops the data.
A width averages the selected points. Interpolation estimates values on a new grid.

(how-to-python-extract-path)=

## Extracting data along a momentum path

Define the path vertices in the coordinates of `momentum_data`, then select a step size
in the same inverse-length units. This example follows Γ–M–K–Γ for the hexagonal model
used by {func}`erlab.io.exampledata.generate_data`:

```python
import numpy as np

import erlab.analysis as era

lattice_constant = 6.97
high_symmetry_vertices = {
    "kx": [
        0.0,
        2 * np.pi / (np.sqrt(3) * lattice_constant),
        2 * np.pi / (np.sqrt(3) * lattice_constant),
        0.0,
    ],
    "ky": [0.0, 0.0, 2 * np.pi / (3 * lattice_constant), 0.0],
}
high_symmetry_cut = era.interpolate.slice_along_path(
    momentum_data,
    vertices=high_symmetry_vertices,
    step_size=0.005,
)
```

The figure below shows the Γ–M–K–Γ path on a constant energy surface and the interpolated
energy–momentum cut. See {doc}`high-symmetry cuts <../plotting/high-symmetry-cuts>` for
the Python plotting code and Figure Composer steps.

```{eval-rst}
.. plot:: how_to/high_symmetry_cuts.py plot_high_symmetry_cut
   :include-source: false
   :alt: Γ–M–K–Γ path on a constant energy surface beside the interpolated energy–momentum cut
```

The result uses `path` as the interpolation dimension and retains `kx` and `ky` as
coordinates along that path. Confirm that the vertices follow the intended reciprocal-
space trajectory before interpreting the result.

See {func}`erlab.analysis.interpolate.slice_along_path` for closed paths, dimension
selection, and explicit sampling points.

(how-to-python-compare-slices)=

## Comparing slices from multidimensional data

Use {func}`erlab.plotting.plot_slices` to select and plot several coordinate values.
Plot independent and shared intensity limits when you must choose between feature
visibility and direct intensity comparison:

```python
import matplotlib.pyplot as plt
import erlab.plotting as eplt

energies = [-0.4, -0.2, 0.0]
fig, axes = plt.subplots(
    2,
    3,
    figsize=(6.4, 4.0),
    layout="compressed",
    sharex=True,
    sharey=True,
)

for row in axes:
    eplt.plot_slices(
        [data],
        eV=energies,
        eV_width=0.05,
        axes=row,
        axis="image",
        gamma=0.5,
        annotate=False,
        cmap="Greys",
    )
    eplt.label_subplot_properties(row, values={"Eb": energies})

eplt.unify_clim(axes[1])
eplt.clean_labels(axes)
axes[0, 0].set_title("Independent intensity limits", loc="left")
axes[1, 0].set_title("Shared intensity limits", loc="left")
```

```{eval-rst}
.. plot:: how_to/inspection_and_selection.py compare_multidimensional_slices
   :include-source: false
   :alt: Three constant energy slices with independent intensity limits above the same slices with shared limits
```

`eV_width` averages each slice over the stated energy width. Remove it when nearest-
coordinate selection is required. The first row retains the automatic limit of each
slice. {func}`erlab.plotting.unify_clim` applies one range to the second row. Use the
shared range when intensity differences between slices must remain visible. Use
independent limits when only feature positions must be compared and each panel needs
its own contrast.

See {func}`erlab.plotting.plot_slices` for layout and normalization arguments.
