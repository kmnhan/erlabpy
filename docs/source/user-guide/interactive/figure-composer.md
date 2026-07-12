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

Open data in ImageTool Manager, then use one of these entry points:

- From ImageTool, right-click an image or line plot and choose {guilabel}`New Figure`.
  - If multiple cursors exist, a figure with multiple axes is created. This is useful
    for quickly generating comparison figures with several slices of the same data.
- From the same menu, choose {guilabel}`Append to Figure` to add the current plot to an
  existing figure.
- From ImageTool Manager, select one or more ImageTool rows and click
  {guilabel}`Add to Figure…` in the right-click context menu of the selection.
- Drag ImageTool rows from the manager into an open Figure Composer window to add them
  as sources.

Figures are listed in the manager's {guilabel}`Figures` tab, which appears after a
figure is created.

(figure-composer-sources)=

## Sources

The {guilabel}`Sources` tab lists the named data variables stored with the figure. Data
must be added to this list before it can be plotted.

Use {guilabel}`Add…` to select ImageTool rows from the manager. You can also drag
ImageTool rows from the manager into the composer controls or figure window.

Select one or more sources and use {guilabel}`Refresh` to update them from their
ImageTools.

When you choose {guilabel}`Add to Figure…` from ImageTool Manager, the corresponding
ImageTool rows are added to the figure sources. In that dialog, different choices for
{guilabel}`Action` determine how the selected data is added to the figure:

- {guilabel}`New Figure` creates another Figure Composer window.
- {guilabel}`Add New Step` adds the selected ImageTool data and appends a plotting step.
- {guilabel}`Add Source Only` adds the selected ImageTool data to the figure without
  creating or changing recipe steps.
- {guilabel}`Replace Source` keeps the existing figure recipe intact but swaps one
  source to use data from the selected ImageTool. This is useful when you have formatted
  a figure and want to reuse the same recipe, axes, styles, and generated variable names
  with updated or comparable data.

(figure-composer-layout)=

## Layout

The {guilabel}`Layout` tab controls the global figure structure. You can define the size
and DPI of the figure, and the number of axes and their arrangement.

- Use {guilabel}`Subplots` mode for regular grids created with
  {func}`matplotlib.pyplot.subplots`.

- Use {guilabel}`GridSpec` mode for more complex figures that include axes that span
  several cells or nested regions created with {class}`matplotlib.gridspec.GridSpec`.
  Drag in the GridSpec editor to create rectangular axes or nested grids. Open a nested
  grid to edit it in place, then use the breadcrumb controls to return to the parent
  grid.

(figure-composer-recipe)=

## Recipe steps

The {guilabel}`Recipe` tab contains a list of steps that generate the figure content.
Each step is an interface to a function or method call that modifies the figure. The
step list is ordered, and the generated code runs in that order, so steps can depend on
the figure state created by earlier steps.

Every step has a type, a target (axes or figure), and a set of controls for the
arguments of the plotting or styling calls it generates.

The step table shows each operation, its target, and its current status. For steps that
act on axes, the {guilabel}`Target` column highlights the affected axes in a miniature
of the current subplot or GridSpec layout. The {guilabel}`Status` reports missing sources, invalid targets or inputs, and rendering errors
when they occur. Hover over a reported problem for details.

There are several step types:

- {guilabel}`Set Palette` to set the line color cycle with {func}`seaborn.set_palette`.
- {guilabel}`Image Plot` for one two-dimensional image on one axes. Uses
  {func}`erlab.plotting.plot_array`.
- {guilabel}`Slice Plot` for plotting multiple slices on multiple axes. Uses
  {func}`erlab.plotting.plot_slices`.
- {guilabel}`Line/Profile` for extracted one-dimensional profiles. This also provides an
  ability to create MDC/EDC stack plots. You can either use this step as a simple
  interface to {meth}`xarray.DataArray.plot` with 1D data, or use it to extract multiple
  profiles from higher dimensional data.
- {guilabel}`BZ Overlay` for in-plane and out-of-plane Brillouin-zone slice overlays
  drawn with {func}`erlab.plotting.plot_in_plane_bz` and
  {func}`erlab.plotting.plot_out_of_plane_bz`.
- {guilabel}`Photon Energy Overlay` for annotating constant photon energies on
  $k_\parallel$-$k_z$ plots using {meth}`xarray.DataArray.kspace.hv_to_kz`.
- {guilabel}`ERLab Method` for a subset of {mod}`erlab.plotting` functions such as
  colorbar and annotation utilities.
- {guilabel}`Axes Method` for a subset of Matplotlib `ax.*` methods.
- {guilabel}`Figure Method` for a subset of Matplotlib `fig.*` methods.
- {guilabel}`Python` for arbitrary code snippets.

### Editing steps

Selecting a step opens its controls, which vary based on the step type. Each control is
an interface to an argument passed onto the underlying function or method.

:::{tip}

Most controls have a tooltip that appears when you hover over them. Some steps have a
button that leads to the relevant documentation webpage for the underlying function or
method.

:::

Use the checkbox beside a step to enable or disable it. Steps can be cut, copied,
pasted, or removed from the toolbar. Reorder steps by dragging their rows. Use the
right-click context menu to duplicate steps or move them.

By selecting multiple steps, you can edit them simultaneously to apply the same change
to all selected steps. Copied or cut steps can be pasted into another Figure Composer.

When the source and destination composers are open in the same app process, pasted steps
also bring the data sources they use.

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

[Matplotlib stylesheets](https://matplotlib.org/stable/users/explain/customizing.html)
are a powerful way to control the styling of all figure elements. Figure Composer
applies the Matplotlib stylesheets configured in the shared {guilabel}`Settings`
window. These stylesheets affect new figure defaults such as size, dpi, and export
settings, as well as the styling of all figure elements.
If {guilabel}`DPI` in Settings has {guilabel}`Override stylesheet` enabled, that value
is used for new figures instead of the stylesheet default.

To add custom Matplotlib stylesheets:

1. Open the shared settings window: on macOS, choose {menuselection}`Preferences…` from
   the application menu next to {fab}`apple` (macOS) or {menuselection}`File -->
   Settings` (Windows/Linux).
2. Select {guilabel}`Figure Composer` in the sidebar. In the {guilabel}`Stylesheets`
   option, click {guilabel}`Open Folder`.
3. Copy `*.mplstyle` files into that folder.
4. Click {guilabel}`Reload` to update the stylesheet list.
5. Add the stylesheet by name. The name shown in settings is the file name without the
   `.mplstyle` suffix.

Saved stylesheet names are kept even if the corresponding style is not currently
available. Unavailable styles remain visible in Settings, are skipped while rendering
and when copying generated code, and become active again after the `.mplstyle` file is
restored and the list is reloaded.

In addition to the default stylesheets provided with Matplotlib, the list includes some
stylesheets provided by ERLab, such as `khan`, `times`, and `nature`. Try them out to
see how they affect the figures.
