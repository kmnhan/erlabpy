(how-to-plotting-photon-energy-annotations)=

# Photon-energy annotations

Use this guide to mark constant-photon-energy curves on converted
$k_\parallel$-$k_z$ intensity. Start with an $h\nu$–dependent scan that follows
{ref}`data-conventions`. Complete {ref}`how-to-python-convert-photon-energy-scan`
before you add the curves.

## Python

Calculate the $k_z$ positions at the binding energy shown in the figure:

```python
photon_energies = [30, 45, 60]
binding_energy = -0.3
kz_values = converted.kspace.hv_to_kz(photon_energies).qsel(
    eV=binding_energy,
)
```

Plot one curve for each photon energy:

```python
import matplotlib.pyplot as plt

import erlab.plotting as eplt

fig, ax = plt.subplots(figsize=(3.4, 3.0), layout="compressed")
eplt.plot_array(
    converted.qsel(eV=binding_energy).T,
    ax=ax,
    cmap="viridis",
    aspect="equal",
)
for index in range(kz_values.sizes["hv"]):
    kz = kz_values.isel(hv=index)
    ax.plot(kz.kx, kz, label=rf"$h\nu={float(kz.hv):g}$ eV")
ax.legend()
```

```{eval-rst}
.. plot:: how_to/momentum_conversion.py annotate_photon_energies
   :include-source: false
   :alt: Converted constant energy surface with calculated kz values for three photon energies
```

The curves use the stored geometry, work function, and inner potential. They do not
show measured intensity at a new photon energy.

## Figure Composer

1. Add `converted` in {guilabel}`Sources`.
2. Use one axes in {guilabel}`Layout`.
3. Add a {guilabel}`Slice Plot` step for `converted`. Set {guilabel}`Dimension` to
   `eV`, enter the required binding energy, and set {guilabel}`Axis` to
   {guilabel}`image`. Enable {guilabel}`Transpose` so that $k_\parallel$ is horizontal
   and $k_z$ is vertical.
4. Add a {guilabel}`Photon Energy Overlay` step after the image. Select `converted` as
   {guilabel}`Overlay data` and target the same axes.
5. Under {guilabel}`hν`, enter the photon energies and the same
   {guilabel}`Binding energy` used for the image.
6. Under {guilabel}`Style`, enable {guilabel}`Legend`. Adjust the line and label
   controls only when the curves obscure the data.

See {meth}`hv_to_kz <xarray.DataArray.kspace.hv_to_kz>` for the coordinate calculation
and {ref}`figure-composer-recipe` for Figure Composer step types.
