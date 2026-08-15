(how-to-plotting)=

# Plotting gallery

Select an example to open its Python and Figure Composer instructions. The Python code
is the standard implementation for each output. For figure creation, export, and recipe
reuse, see {doc}`the Figure Composer guide <../gui/plotting>`.

```{toctree}
:hidden:

colorbars
maps-and-cuts
two-dimensional-colormaps
annotations
titles-and-labels
axis-units
core-levels
brillouin-zones
brillouin-zone-overlays
out-of-plane-brillouin-zones
figure-styles
```

## Data views

:::::{grid} 1 1 2 2
:gutter: 2

::::{grid-item-card} Colorbars
:link: colorbars
:link-type: doc
:class-card: plotting-gallery-card

```{eval-rst}
.. plot:: how_to/plotting.py add_intensity_colorbar
   :include-source: false
   :show-source-link: false
   :class: plotting-gallery-preview
   :alt: Energy momentum intensity plot with a colorbar
```
::::

::::{grid-item-card} Maps and cuts
:link: maps-and-cuts
:link-type: doc
:class-card: plotting-gallery-card

```{eval-rst}
.. plot:: how_to/plotting.py combine_maps_and_cuts
   :include-source: false
   :show-source-link: false
   :class: plotting-gallery-preview
   :alt: Constant energy maps and energy momentum cuts
```
::::

::::{grid-item-card} 2D colormaps
:link: two-dimensional-colormaps
:link-type: doc
:class-card: plotting-gallery-card

```{eval-rst}
.. plot:: how_to/plotting.py plot_intensity_and_asymmetry
   :include-source: false
   :show-source-link: false
   :class: plotting-gallery-preview
   :alt: Two-dimensional colormap of intensity and asymmetry
```
::::

:::::

## Annotations

:::::{grid} 1 1 2 2
:gutter: 2

::::{grid-item-card} ARPES cut annotations
:link: annotations
:link-type: doc
:class-card: plotting-gallery-card

```{eval-rst}
.. plot:: how_to/plotting.py annotate_arpes_figure
   :include-source: false
   :show-source-link: false
   :class: plotting-gallery-preview
   :alt: ARPES cut with a Fermi level line, momentum labels, and a panel label
```
::::

::::{grid-item-card} Titles and axis labels
:link: titles-and-labels
:link-type: doc
:class-card: plotting-gallery-card

```{eval-rst}
.. plot:: how_to/plotting.py set_panel_titles_and_labels
   :include-source: false
   :show-source-link: false
   :class: plotting-gallery-preview
   :alt: Constant energy map and energy distribution curve with axis labels
```
::::

::::{grid-item-card} Energy in meV
:link: axis-units
:link-type: doc
:class-card: plotting-gallery-card

```{eval-rst}
.. plot:: how_to/plotting.py display_energy_in_mev
   :include-source: false
   :show-source-link: false
   :class: plotting-gallery-preview
   :alt: Energy momentum cut with energy in millielectronvolts
```
::::

::::{grid-item-card} Annotating core levels
:link: core-levels
:link-type: doc
:class-card: plotting-gallery-card

```{eval-rst}
.. plot:: how_to/plotting.py mark_core_levels
   :include-source: false
   :show-source-link: false
   :class: plotting-gallery-preview
   :alt: Core level spectrum with bismuth and selenium reference lines
```
::::

:::::

## Brillouin zones

:::::{grid} 1 1 2 2
:gutter: 2

::::{grid-item-card} 2D Brillouin zones
:link: brillouin-zones
:link-type: doc
:class-card: plotting-gallery-card

```{eval-rst}
.. plot:: how_to/plotting.py draw_two_dimensional_brillouin_zone
   :include-source: false
   :show-source-link: false
   :class: plotting-gallery-preview
   :alt: Hexagonal first Brillouin zone
```
::::

::::{grid-item-card} Brillouin-zone overlays
:link: brillouin-zone-overlays
:link-type: doc
:class-card: plotting-gallery-card

```{eval-rst}
.. plot:: how_to/plotting.py overlay_brillouin_zone
   :include-source: false
   :show-source-link: false
   :class: plotting-gallery-preview
   :alt: Constant energy map with a Brillouin zone boundary
```
::::

::::{grid-item-card} Out-of-plane Brillouin-zone sections
:link: out-of-plane-brillouin-zones
:link-type: doc
:class-card: plotting-gallery-card

```{eval-rst}
.. plot:: how_to/plotting.py draw_out_of_plane_brillouin_zone
   :include-source: false
   :show-source-link: false
   :class: plotting-gallery-preview
   :alt: Out-of-plane Brillouin zone slice with marked vertices
```
::::

:::::

## Styles

:::::{grid} 1 1 2 2
:gutter: 2

::::{grid-item-card} Figure styles
:link: figure-styles
:link-type: doc
:class-card: plotting-gallery-card

:::{div} plotting-gallery-symbol
{material-regular}`palette;5rem`
:::
::::

:::::
