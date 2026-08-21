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
cut-trajectories
high-symmetry-cuts
annotations
photon-energy-annotations
titles-and-labels
axis-units
core-levels
brillouin-zones
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

::::{grid-item-card} Polygon masking
:link: how-to-python-mask-polygon
:link-type: ref
:class-card: plotting-gallery-card

```{eval-rst}
.. plot:: how_to/inspection_and_selection.py mask_momentum_region
   :include-source: false
   :show-source-link: false
   :class: plotting-gallery-preview
   :alt: Constant energy map with a polygon boundary and the resulting masked map
```
::::

:::::

## Momentum-space plots

:::::{grid} 1 1 2 2
:gutter: 2

::::{grid-item-card} Cut trajectories
:link: cut-trajectories
:link-type: doc
:class-card: plotting-gallery-card

```{eval-rst}
.. plot:: how_to/momentum_conversion.py overlay_cut_path
   :include-source: false
   :show-source-link: false
   :class: plotting-gallery-preview
   :alt: Angular cut trajectory on a converted constant energy surface
```
::::

::::{grid-item-card} hν annotations
:link: photon-energy-annotations
:link-type: doc
:class-card: plotting-gallery-card

```{eval-rst}
.. plot:: how_to/momentum_conversion.py annotate_photon_energies
   :include-source: false
   :show-source-link: false
   :class: plotting-gallery-preview
   :alt: Constant-photon-energy curves on converted momentum-space intensity
```
::::

::::{grid-item-card} High-symmetry cuts
:link: high-symmetry-cuts
:link-type: doc
:class-card: plotting-gallery-card

```{eval-rst}
.. plot:: how_to/inspection_and_selection.py plot_high_symmetry_cut
   :include-source: false
   :show-source-link: false
   :class: plotting-gallery-preview
   :alt: Γ–M–K–Γ path and interpolated energy–momentum cut
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

::::{grid-item-card} First Brillouin zone
:link: how-to-plotting-first-brillouin-zone
:link-type: ref
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
:link: how-to-python-overlay-brillouin-zone
:link-type: ref
:class-card: plotting-gallery-card

```{eval-rst}
.. plot:: how_to/plotting.py overlay_brillouin_zone
   :include-source: false
   :show-source-link: false
   :class: plotting-gallery-preview
   :alt: Constant energy map with a Brillouin zone boundary
```
::::

::::{grid-item-card} In-plane sections
:link: how-to-plotting-in-plane-brillouin-zone-sections
:link-type: ref
:class-card: plotting-gallery-card

```{eval-rst}
.. plot:: how_to/plotting.py draw_in_plane_brillouin_zone
   :include-source: false
   :show-source-link: false
   :class: plotting-gallery-preview
   :alt: In-plane Brillouin zone section with marked vertices
```
::::

::::{grid-item-card} Out-of-plane Brillouin-zone sections
:link: how-to-plotting-out-of-plane-brillouin-zones
:link-type: ref
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
