# Publication-quality ARPES plots

Build the final figure from accepted calibrated data and validated fit results. Keep the
figure code in the notebook and save from explicit figure objects.

## Contents

- [Choose the data representation](#choose-the-data-representation)
- [Plot cuts and constant-energy maps](#plot-cuts-and-constant-energy-maps)
- [Plot fitted dispersions](#plot-fitted-dispersions)
- [Apply publication styling](#apply-publication-styling)
- [Export and inspect](#export-and-inspect)

## Choose the data representation

- Show raw or minimally processed intensity for primary evidence.
- Show derivative, curvature, normalized, or symmetrized data as processed data. Label
  the operation and keep a raw comparison nearby.
- Average a documented physical energy or momentum window before differentiation when
  one-pixel slices are noisy. The window sizes used must be explitly provided to the
  user. Apply coordinate-aware smoothing before curvature or derivative processing. Do
  not publish a processed panel that is visibly dominated by pixel noise.
- Use normalization to reduce broad detector gain or matrix-element modulation only
  when the selected normalization region is visible and documented. Do not interpret a
  normalized intensity change as spectral-weight transfer without separate evidence.
- Use a sequential colormap for intensity and a diverging colormap centered at zero for
  signed differences or residuals.
- Use identical color limits when panels are compared quantitatively.
- Do not use gamma or nonlinear normalization to hide background or weak contrary
  evidence. Record any nonlinear normalization in code.
- Preserve physical axis directions and units. Mark the Fermi level and high-symmetry
  points explicitly.
- Crop or mask detector gaps and analyzer-acceptance boundaries before derivatives and
  contour extraction. Keep a raw panel that shows the excluded region.

When comparing samples, doping, terminations, or photon energies, separate band-position
comparisons from intensity comparisons. Match photon energy, polarization, analyzer
mode, integration widths, smoothing widths, and color normalization before comparing
intensity. If these conditions differ, use normalized panels only for band geometry and
state that matrix elements prevent a quantitative intensity comparison.

## Plot cuts and constant-energy maps

Use a finite energy width for a constant-energy map. Put the center and full integration
width in the notebook configuration:

```python
constant_energy_map = momentum_data.qsel(
    eV=map_energy,
    eV_width=map_energy_width,
)

fig, ax = plt.subplots(figsize=(3.2, 3.0), layout="compressed")
eplt.plot_array(
    constant_energy_map,
    ax=ax,
    cmap="Greys",
    gamma=0.7,
    aspect="equal",
    colorbar=True,
)
ax.set_title(f"$E - E_F = {map_energy:.3f}$ eV")
```

For several energies, use `plot_slices` and one consistent normalization:

```python
fig, axes = eplt.plot_slices(
    [momentum_data],
    eV=map_energies,
    eV_width=map_energy_width,
    axis="image",
    cmap="Greys",
    gamma=0.7,
    figsize=(6.5, 2.2),
)
eplt.unify_clim(axes)
eplt.clean_labels(axes)
eplt.label_subplot_properties(axes, values={"Eb": map_energies})
```

Treat a constant-energy contour as a constant-energy intensity map. Do not add
Matplotlib contour lines unless the user explicitly requests them. Add
`plot_in_plane_bz` only with verified lattice vectors and orientation.

For an energy-momentum cut:

```python
fig, ax = plt.subplots(figsize=(3.4, 2.8), layout="compressed")
eplt.plot_array(momentum_cut, ax=ax, cmap="Greys", gamma=0.6, colorbar=True)
eplt.fermiline(ax=ax, color="tab:red", lw=0.8, ls="--")
```

## Plot fitted dispersions

Overlay only accepted fit centers. Preserve gaps where fits were rejected:

```python
branch_styles = {
    "p0_center": ("left branch", "tab:blue"),
    "p1_center": ("right branch", "tab:orange"),
}
for parameter in valid_centers.param.values:
    label, color = branch_styles[str(parameter)]
    center = valid_centers.sel(param=parameter)
    stderr = peak_stderr.sel(param=parameter).where(center.notnull())
    ax.errorbar(
        center,
        center.eV,
        xerr=stderr,
        fmt=".",
        ms=2.5,
        lw=0.6,
        color=color,
        label=label,
    )
```

Plot representative fit curves and residuals in a separate validation figure. Do not
make the final overlay the only fit evidence.

For a path through high-symmetry points, calculate cumulative path distance from the
actual momentum coordinates. Place ticks at segment boundaries and draw light vertical
guides. Use mathematical symbols such as `Γ` in visible labels.

## Apply publication styling

Start with ERLab styles and adapt the figure to the target journal:

```python
with plt.style.context(["erlab.general", "erlab.nature"]):
    fig, ax = plt.subplots(figsize=(3.4, 2.6), layout="compressed")
    eplt.plot_array(momentum_cut, ax=ax, cmap="Greys", gamma=0.6)
    eplt.fermiline(ax=ax, lw=0.8, ls="--")
```

Add a font style such as `erlab.helvetica`, `erlab.arial`, or `erlab.times` only when
the font is installed and the target format permits it.

Check:

- final printed width and height;
- readable text at that size;
- consistent line widths, marker sizes, and panel labels;
- one colorbar per shared scale rather than repeated redundant colorbars;
- units on every physical axis and colorbar where meaningful;
- no clipped annotations or duplicate labels;
- accessible contrast in print and on screen;
- legends that identify physical branches rather than internal parameter names.

When a legend or panel label is on a dark intensity image, use a contrasting opaque or
semi-opaque background, use a suitable text color, or move it outside the image. Keep
panel labels clear of legends, titles, and plotted data.

Use `clean_labels`, `label_subplots`, `label_subplot_properties`, `nice_colorbar`,
`unify_clim`, `fermiline`, and `mark_points` when they directly simplify the figure.

## Export and inspect

Save vector output for text, axes, and line art. Also save a high-resolution raster when
the journal or collaboration needs it:

```python
pdf_path = output_directory / "band_dispersion.pdf"
png_path = output_directory / "band_dispersion.png"

fig.savefig(pdf_path)
fig.savefig(png_path, dpi=600)
print(pdf_path.resolve())
print(png_path.resolve())
```

After saving:

1. Confirm that both files exist and are nonempty.
2. Render or open the saved figure rather than trusting the inline display.
3. Inspect labels, fonts, clipping, colorbar limits, error bars, and image resolution.
4. Confirm that the exported figure uses the accepted calibration and fit mask.
5. Record the output paths and any processing used for each panel.

Do not overwrite an existing user figure unless the user requested that path. Use a
descriptive new filename by default.
