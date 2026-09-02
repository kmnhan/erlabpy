(figure-composer)=

# Figure Composer

Figure Composer is a GUI for building publication-quality Matplotlib figures from
ImageTool data, without needing to write or edit code. It is designed for the common
workflow where you explore data interactively, then turn the useful view into a
reproducible figure without rewriting the whole plot in a notebook.

The composer stores a figure as a *recipe*: a layout of axes plus an ordered list of
steps that each correspond to a plotting operation.

(figure-composer-open)=

For creating a figure and managing sources, see {ref}`how-to-gui-figure-composer`.
For complete Python and Figure Composer examples, see the
{ref}`plotting gallery <how-to-plotting>`.

(figure-composer-sources)=

## Sources

Figure Composer can start with or without data sources. In ImageTool Manager,
{menuselection}`File --> New Empty Figure` creates a figure without sources or recipe
steps. Use {guilabel}`Add…` in the {guilabel}`Sources` tab, or drag ImageTool rows from
the Manager, to add sources later.

The {guilabel}`Sources` tab lists the named data variables stored with the figure. Use
{guilabel}`Refresh` to update selected sources from their ImageTools. Use
{guilabel}`Reveal in Manager` to select their ImageTool rows and bring the Manager to
the front.

The {guilabel}`Add to Figure…` action in ImageTool Manager can create a figure, add a
plotting step, add a source without changing the recipe, or replace a source. See
{ref}`how-to-gui-reuse-figure-recipe` for the replacement procedure.

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
  grid. Select an axes region, then use the x or y control under {guilabel}`Share axes`
  to select the axes that share that coordinate axis.

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

- {guilabel}`Set Palette` to set the line color cycle with a named palette, custom
  colors, or a generated seaborn cubehelix, diverging, light, or dark palette.
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
  colorbar and annotation utilities. This includes
  {func}`erlab.plotting.plot_core_levels` for expected core-level energies.
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

Use the toolbar in the figure window to change the plot. The subplot configuration
button edits subplot spacing and the layout engine. The axes customization button edits
the selected axes.

(figure-composer-reproducibility)=

For preserving a figure recipe, see {ref}`how-to-gui-preserve-figure-recipe`.

(figure-composer-options)=

For export procedures, see {ref}`how-to-gui-export-figure`.

(figure-composer-styles)=

## Styles and export settings

Figure Composer applies the Matplotlib stylesheets selected in the shared Settings
window. These styles control new figure defaults and the appearance of figure elements.
An enabled {guilabel}`Override stylesheet` setting takes precedence over the stylesheet
value for the corresponding option.

The {guilabel}`Export` tab controls DPI, transparency, bounding box, and padding for one
figure. {guilabel}`Use Defaults` inherits the current workspace or user setting.

Custom `*.mplstyle` files are discovered from the Figure Composer stylesheet folder.
Saved style names remain in settings when their files are unavailable. Missing styles
are skipped during rendering and code generation, then become active again after the
file is restored and the style list is reloaded.
