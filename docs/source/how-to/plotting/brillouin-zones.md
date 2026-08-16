(how-to-plotting-two-dimensional-bz)=

(how-to-python-draw-two-dimensional-bz)=

# 2D Brillouin zones

## Python

Construct the real-space lattice vectors for the crystal. Pass them to
{func}`erlab.plotting.plot_bz`:

```python
import matplotlib.pyplot as plt

import erlab
import erlab.plotting as eplt

avec = erlab.lattice.abc2avec(3.0, 3.0, 5.0, 90.0, 90.0, 120.0)

fig, ax = plt.subplots(figsize=(2.5, 2.5), layout="compressed")
eplt.plot_bz(avec, ax=ax)
ax.set(
    xlabel=r"$k_x$ (Å$^{-1}$)",
    ylabel=r"$k_y$ (Å$^{-1}$)",
    xlim=(-1.5, 1.5),
    ylim=(-1.5, 1.5),
    aspect="equal",
)
```

```{eval-rst}
.. plot:: how_to/plotting.py draw_two_dimensional_brillouin_zone
   :include-source: false
   :alt: Hexagonal first Brillouin-zone polygon on equal reciprocal-momentum axes
```

## Figure Composer

The current {guilabel}`BZ Overlay` step draws an in-plane section. It does not draw the
single first-zone polygon in this guide.

```{include} ../../_includes/figure-composer-planned-step.md
```

1. In ImageTool Manager, choose
   {menuselection}`File --> New Empty Figure`.
2. Set {guilabel}`Layout` to a $1 \times 1$ grid.
3. Add a {guilabel}`Python` step.
4. Review this code, enter it in {guilabel}`Code`, and enable {guilabel}`Trusted`:

```python
import erlab

avec = erlab.lattice.abc2avec(3.0, 3.0, 5.0, 90.0, 90.0, 120.0)
eplt.plot_bz(avec, ax=ax)
ax.set(
    xlabel=r"$k_x$ (Å$^{-1}$)",
    ylabel=r"$k_y$ (Å$^{-1}$)",
    aspect="equal",
)
```

Replace the lattice parameters with those of the measured material. See
{func}`erlab.plotting.plot_bz` for reciprocal input, rotation, and offset arguments.
