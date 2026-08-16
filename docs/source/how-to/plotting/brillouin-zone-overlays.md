(how-to-plotting-brillouin-zone-overlays)=

(how-to-python-overlay-brillouin-zone)=

# Brillouin-zone overlays

## Python

For a hexagonal material, use its measured in-plane lattice constant to draw the
boundary over a constant energy map:

```python
import matplotlib.pyplot as plt

import erlab.plotting as eplt

lattice_constant = 6.97

fig, ax = plt.subplots(figsize=(3.4, 3.0), layout="compressed")
eplt.plot_array(
    constant_energy_map,
    ax=ax,
    cmap="Greys",
    gamma=0.5,
    aspect="equal",
)
eplt.plot_hex_bz(
    a=lattice_constant,
    ax=ax,
    fill=False,
    edgecolor="tab:purple",
    linestyle="--",
    linewidth=1.2,
)
```

```{eval-rst}
.. plot:: how_to/plotting.py overlay_brillouin_zone
   :include-source: false
   :alt: Constant energy map with an in-plane Brillouin zone boundary
```

## Figure Composer

1. Set {guilabel}`Layout` to a $1 \times 1$ grid.
2. Add `constant_energy_map` in {guilabel}`Sources`.
3. Add an {guilabel}`Image Plot` step and set {guilabel}`Aspect` to `equal`.

The current {guilabel}`BZ Overlay` step draws an in-plane section. It does not draw the
single hexagonal boundary used here.

```{include} ../../_includes/figure-composer-planned-step.md
```

4. Add a {guilabel}`Python` step after the image.
5. Review this code, enter it in {guilabel}`Code`, and enable {guilabel}`Trusted`:

```python
eplt.plot_hex_bz(
    a=6.97,
    ax=ax,
    fill=False,
)
```

Replace `a` with the in-plane lattice constant of the measured material. The boundary
must use the correct lattice parameters and orientation. Do not use the appearance of
the intensity to select or resize the boundary. See
{func}`erlab.plotting.plot_hex_bz` for rotation and offset arguments.
