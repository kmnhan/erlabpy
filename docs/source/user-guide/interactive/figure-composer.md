(figure-composer)=

# Figure Composer

Figure Composer is a GUI for building publication-quality Matplotlib figures from
ImageTool data, without needing to write or edit code. It is designed for the common
workflow where you explore data interactively, then turn the useful view into a
reproducible figure without rewriting the whole plot in a notebook.

The composer stores a figure as a *recipe*: a layout of axes plus an ordered list of
steps that each correspond to a plotting operation.

(figure-composer-open)=

## Opening Figure Composer

There are three main entry points:

- From ImageTool, right-click an image or line plot and choose {guilabel}`New Figure`.
  This creates a new figure seeded from the selected image.
  - If multiple cursors exist, a figure with multiple axes is created. This is useful
    for quickly generating comparison figures with several slices of the same data.
- From ImageTool, choose {guilabel}`Append to Figure` from the same context menu to add
  the current plot to an existing figure.
- From the manager, select one or more ImageTool rows and click
  {guilabel}`Add to Figure…` in the right-click context menu of the selection.

Figures are listed in the manager's {guilabel}`Figures` tab, which is hidden until a
figure is created.

A Figure Composer is comprised of two main windows: the window that contains all the
controls, and the Matplotlib figure window that shows the rendered figure.

(figure-composer-layout)=

## Layout

In the controls window of the Figure Composer, the layout shows tabs for the global
figure structure and the recipe steps.

The {guilabel}`Layout` tab controls the global figure structure, where you can define
the size and DPI of the figure, and the number of axes and their arrangement.

Use {guilabel}`Subplots` mode for regular grids created with
{func}`matplotlib.pyplot.subplots`. Set rows, columns, shared x/y axes, layout engine,
and optional width or height ratios. This is the quickest way to create a row of
constant-energy maps or a stacked image/profile figure.

Use {guilabel}`GridSpec` mode when the figure needs axes that span several cells or
nested regions created with {class}`matplotlib.gridspec.GridSpec` Drag in the GridSpec
editor to create rectangular axes or nested grids. Open a nested grid to edit it in
place, then use the breadcrumb controls to return to the parent grid. Axis labels are
optional; the composer generates stable names for code when labels are left blank.

Changing the layout does not silently rewrite step targets. If a step points to axes
that no longer exist, the step remains editable and is marked invalid until you repair
the target axes.

(figure-composer-recipe)=

## Recipe steps

The {guilabel}`Recipe` tab contains a list of steps that generate the figure content.
Each step is an interface to a function or method call that modifies the figure. The
step list is ordered, and the generated code runs in that order, so steps can depend on
the figure state created by earlier steps.

Every step has a type, a target (axes or figure), and a set of controls for the
arguments of the plotting or styling calls it generates.

There are several step types, each with a different set of controls for the generated code:

- {guilabel}`Plot Slices` as an interface to {func}`erlab.plotting.plot_slices`.
- {guilabel}`Line/Profile` for extracted one-dimensional profiles. This also provides an
  ability to create MDC/EDC stack plots. You can either use this step as a simple
  interface to {meth}`xarray.DataArray.plot` with 1D data, or use it to extract multiple
  profiles from higher dimensional data.
- {guilabel}`BZ Overlay` for in-plane and out-of-plane Brillouin-zone slice overlays
  drawn with {func}`erlab.plotting.plot_in_plane_bz` and
  {func}`erlab.plotting.plot_out_of_plane_bz`.
- {guilabel}`ERLab Method` for a subset of {mod}`erlab.plotting` functions such as
  colorbar and annotation utilities.
- {guilabel}`Axes Method` for a subset of Matplotlib `ax.*` methods.
- {guilabel}`Figure Method` for a subset of Matplotlib `fig.*` methods.
- {guilabel}`Custom Code` for arbitrary code snippets.

The {guilabel}`BZ Overlay` step stores conventional-cell lattice parameters,
centering, slice mode, angle, normalized `kz` in units of `pi/c`, optional bounds, and
line or point styling. When code is copied, the recipe explicitly constructs the real
lattice vectors, converts non-primitive cells to primitive vectors when needed, converts
to reciprocal vectors, and calls the public plotting helper for the selected slice.
Figures created from ktool converted outputs copy the current ktool BZ settings when the
ktool overlay is enabled; ordinary momentum-cut ImageTool data can seed the slice mode
and bounds from `kx`, `ky`, and `kz` coordinates.

### Editing steps

Selecting a step opens its controls, which vary based on the step type. Each control is
an interface to an argument passed onto the underlying function or method.

:::{tip}

Most controls have a tooltip that appears when you hover over them. Some steps have a
button that leads to the relevant documentation webpage for the underlying function or
method.

:::

Steps can be enabled, disabled, duplicated, reordered, cut, copied, pasted, or removed.

By selecting multiple steps, you can edit them simultaneously to apply the same change
to all selected steps. Copied or cut steps can be pasted into another Figure Composer.

When the source and destination composers are open in the same app process, pasted steps
also bring the data sources they use.

(figure-composer-sources)=

### Step sources

Because data sources are selected and updated per recipe step, the step controls include
a {guilabel}`Sources` view. It lists the data stored with the figure, indicates which
sources are used by the selected step, and lets you update the data available to recipe
steps without rebuilding the rest of the figure.

When you choose {guilabel}`Add to Figure…` from ImageTool Manager, the
{guilabel}`Action` menu updates this same source set:

- {guilabel}`New Figure` creates another Figure Composer window.
- {guilabel}`Add New Step` adds the selected ImageTool data and appends a plotting step.
- {guilabel}`Add Source Only` adds the selected ImageTool data to the figure without
  creating or changing recipe steps.
- {guilabel}`Replace Source` keeps the existing figure recipe intact but swaps one source
  to use data from the selected ImageTool. This is useful when you have formatted a
  figure and want to reuse the same recipe, axes, styles, and generated variable names
  with updated or comparable data.

The {guilabel}`Sources` view also provides refresh controls. Click the row button next
to one source to refresh only that source from its linked open ImageTool, or click
{guilabel}`Refresh Sources` to refresh every source in the figure that is still linked
to an open ImageTool.

(figure-composer-toolbar)=

## Toolbar controls

In addition to the recipe controls, plots can also be customized with the toolbar of the
figure window. You can edit the subplot spacing and edit various figure elements from
dialogs opened from the toolbar buttons.

(figure-composer-reproducibility)=

## Reproducibility

Figure recipes are saved in ImageTool Manager workspace files, so they are portable and
can be shared with collaborators.

Removing a source ImageTool does not delete figures that reference it; the figure is
kept and the source is marked missing until it can be restored or replaced.

Use {guilabel}`Copy Code` in the composer for recipe code that assumes the figure
sources already exist as Python variables.

Use the manager side panel's {guilabel}`Copy Full Code` action when you want recursive
replay code that includes all analysis steps.

(figure-composer-options)=

## Stylesheets and export

Figure Composer applies the Matplotlib stylesheets configured in the interactive
settings dialog in {guilabel}`File → Settings`. These stylesheets affect new figure
defaults such as size, dpi, and export settings, as well as the styling of all figure
elements. If a saved stylesheet is unavailable on the current computer, it remains
visible in settings but is skipped when rendering or generating code.
