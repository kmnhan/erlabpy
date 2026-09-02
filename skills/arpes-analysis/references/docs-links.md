# ERLabPy documentation links

Use these stable pages only when exact API behavior or a public citation is needed.
Prefer the documentation that matches the installed ERLabPy version when it is
available locally.

## Contents

- [Core guides](#core-guides)
- [Optional GUI guides](#optional-gui-guides)
- [Python APIs](#python-apis)

## Core guides

- [First notebook tutorial](https://erlabpy.readthedocs.io/en/stable/tutorials/python/)
- [Data loading and saving](https://erlabpy.readthedocs.io/en/stable/how-to/python/loading-and-saving.html)
- [Data inspection and selection](https://erlabpy.readthedocs.io/en/stable/how-to/python/inspection-and-selection.html)
- [Momentum conversion](https://erlabpy.readthedocs.io/en/stable/how-to/python/momentum-conversion.html)
- [Fermi edge correction](https://erlabpy.readthedocs.io/en/stable/how-to/python/fermi-edge-correction.html)
- [Curve fitting](https://erlabpy.readthedocs.io/en/stable/how-to/python/curve-fitting.html)
- [Plotting gallery](https://erlabpy.readthedocs.io/en/stable/how-to/plotting/index.html)
- [Fermi edge correction](https://erlabpy.readthedocs.io/en/stable/how-to/python/fermi-edge-correction.html)
- [Annotating core levels](https://erlabpy.readthedocs.io/en/stable/how-to/plotting/core-levels.html)
- [Transformations and filtering](https://erlabpy.readthedocs.io/en/stable/how-to/python/transformations-and-filtering.html)
- [ARPES data conventions](https://erlabpy.readthedocs.io/en/stable/explanation/data-conventions.html)
- [Momentum conversion explanation](https://erlabpy.readthedocs.io/en/stable/explanation/momentum-conversion.html)
- [Reference](https://erlabpy.readthedocs.io/en/stable/reference.html)
- [Stable documentation search](https://erlabpy.readthedocs.io/en/stable/search.html)
- [Stable LLM sitemap](https://erlabpy.readthedocs.io/en/stable/llms.txt)
- [Stable LLM export without changelog](https://erlabpy.readthedocs.io/en/stable/llms-full-no-changelog.txt)

## Optional GUI guides

- [ImageTool Manager tutorial](https://erlabpy.readthedocs.io/en/stable/tutorials/index.html#manager-tutorial)
- [Python and GUI workflows](https://erlabpy.readthedocs.io/en/stable/explanation/python-and-gui-workflows.html)
- [Derived results and workspaces](https://erlabpy.readthedocs.io/en/stable/how-to/gui/derived-results-and-workspaces.html)
- [Fermi edge correction (ImageTool Manager)](https://erlabpy.readthedocs.io/en/stable/how-to/gui/fermi-edge-correction.html)
- [ImageTool](https://erlabpy.readthedocs.io/en/stable/reference/gui/imagetool.html)
- [ImageTool Manager](https://erlabpy.readthedocs.io/en/stable/reference/gui/manager.html)
- [Manager extensions](https://erlabpy.readthedocs.io/en/stable/how-to/gui/extensions.html)
- [Manager extension reference](https://erlabpy.readthedocs.io/en/stable/reference/gui/extensions.html)
- [Synchronizing a notebook variable with ImageTool](https://erlabpy.readthedocs.io/en/stable/how-to/gui/python-integration.html#how-to-gui-watch-notebook-variables)
- [Figure Composer](https://erlabpy.readthedocs.io/en/stable/reference/gui/figure-composer.html)
- [Interactive tools](https://erlabpy.readthedocs.io/en/stable/reference/gui/tools/index.html)

## Python APIs

- [`erlab.io.load`](https://erlabpy.readthedocs.io/en/stable/erlab.io.html#erlab.io.load)
- [`xarray.DataArray.qsel`](https://erlabpy.readthedocs.io/en/stable/accessors/xarray.DataArray.qsel.html)
- [`xarray.DataArray.qplot`](https://erlabpy.readthedocs.io/en/stable/accessors/xarray.DataArray.qplot.html)
- [`xarray.DataArray.kspace.convert`](https://erlabpy.readthedocs.io/en/stable/accessors/xarray.DataArray.kspace.convert.html)
- [`xarray.DataArray.kspace.convert_coords`](https://erlabpy.readthedocs.io/en/stable/accessors/xarray.DataArray.kspace.convert_coords.html)
- [`xarray.DataArray.kspace.hv_to_kz`](https://erlabpy.readthedocs.io/en/stable/accessors/xarray.DataArray.kspace.hv_to_kz.html)
- [`xarray.DataArray.kspace.offsets`](https://erlabpy.readthedocs.io/en/stable/accessors/xarray.DataArray.kspace.offsets.html)
- [`xarray.DataArray.kspace.set_normal`](https://erlabpy.readthedocs.io/en/stable/accessors/xarray.DataArray.kspace.set_normal.html)
- [`xarray.DataArray.kspace.set_normal_like`](https://erlabpy.readthedocs.io/en/stable/accessors/xarray.DataArray.kspace.set_normal_like.html)
- [`erlab.analysis.gold.edge`](https://erlabpy.readthedocs.io/en/stable/generated/erlab.analysis.gold.html#erlab.analysis.gold.edge)
- [`erlab.analysis.gold.guess_edge_fit_range`](https://erlabpy.readthedocs.io/en/stable/generated/erlab.analysis.gold.html#erlab.analysis.gold.guess_edge_fit_range)
- [`erlab.analysis.gold.poly`](https://erlabpy.readthedocs.io/en/stable/generated/erlab.analysis.gold.html#erlab.analysis.gold.poly)
- [`erlab.analysis.gold.quick_fit`](https://erlabpy.readthedocs.io/en/stable/generated/erlab.analysis.gold.html#erlab.analysis.gold.quick_fit)
- [`erlab.analysis.gold.correct_with_edge`](https://erlabpy.readthedocs.io/en/stable/generated/erlab.analysis.gold.html#erlab.analysis.gold.correct_with_edge)
- [`MultiPeakModel`](https://erlabpy.readthedocs.io/en/stable/generated/erlab.analysis.fit.models.html#erlab.analysis.fit.models.MultiPeakModel)
- [`FermiEdgeModel`](https://erlabpy.readthedocs.io/en/stable/generated/erlab.analysis.fit.models.html#erlab.analysis.fit.models.FermiEdgeModel)
- [`erlab.plotting.plot_array`](https://erlabpy.readthedocs.io/en/stable/erlab.plotting.html#erlab.plotting.plot_array)
- [`erlab.plotting.plot_slices`](https://erlabpy.readthedocs.io/en/stable/erlab.plotting.html#erlab.plotting.plot_slices)
- [`erlab.plotting.fermiline`](https://erlabpy.readthedocs.io/en/stable/erlab.plotting.html#erlab.plotting.fermiline)
- [`erlab.plotting.plot_core_levels`](https://erlabpy.readthedocs.io/en/stable/erlab.plotting.html#erlab.plotting.plot_core_levels)
- [`erlab.plotting.plot_in_plane_bz`](https://erlabpy.readthedocs.io/en/stable/erlab.plotting.html#erlab.plotting.plot_in_plane_bz)
- [`erlab.plotting.clean_labels`](https://erlabpy.readthedocs.io/en/stable/erlab.plotting.html#erlab.plotting.clean_labels)
- [`erlab.plotting.label_subplot_properties`](https://erlabpy.readthedocs.io/en/stable/erlab.plotting.html#erlab.plotting.label_subplot_properties)
- [`erlab.plotting.label_subplots`](https://erlabpy.readthedocs.io/en/stable/erlab.plotting.html#erlab.plotting.label_subplots)
- [`erlab.plotting.mark_points`](https://erlabpy.readthedocs.io/en/stable/erlab.plotting.html#erlab.plotting.mark_points)
- [`erlab.plotting.nice_colorbar`](https://erlabpy.readthedocs.io/en/stable/erlab.plotting.html#erlab.plotting.nice_colorbar)
- [`erlab.plotting.unify_clim`](https://erlabpy.readthedocs.io/en/stable/erlab.plotting.html#erlab.plotting.unify_clim)
- [`xarray.DataArray.xlm.modelfit`](https://xarray-lmfit.readthedocs.io/stable/accessors/xarray.DataArray.xlm.modelfit.html)
