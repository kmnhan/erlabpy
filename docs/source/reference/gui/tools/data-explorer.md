(guide-data-explorer)=

# Data explorer

```{image} ../../../images/explorer_light.png
:align: center
:alt: Data explorer window in light mode
:class: only-light
```

:::{only} format_html

```{image} ../../../images/explorer_dark.png
:align: center
:alt: Data explorer window in dark mode
:class: only-dark
```

:::

Provides a file-browser-like interface for exploring and visualizing ARPES data stored on
your disk.

The loader list includes general xarray HDF5 (`.h5`), NetCDF (`.nc`, `.nc4`, and
`.cdf`), Zarr (`.zarr`), and supported Igor (`.ibw` and `.pxt`) files.

For most day-to-day browsing and loading, this is faster than the interactive summary
table described in the {ref}`I/O guide <io-summarizing-data>`. Use the summary table when
you want the overview as a DataFrame inside Python or when you are implementing loaders.

You can open the explorer in several ways:

1. From the {ref}`ImageTool manager <imagetool-manager-data-explorer>` with
   {menuselection}`File --> Data Explorer` or {kbd}`Ctrl+E`.
2. From Python with {func}`erlab.interactive.data_explorer`.
3. From the command line with `python -m erlab.interactive.explorer`.

Browsing directories and previewing metadata works in the standalone explorer. Opening
selected files into ImageTool analysis requires a running ImageTool manager. Launching the
explorer from the manager is therefore the recommended path for users who want
to find a dataset and start working with it immediately.
