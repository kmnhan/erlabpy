# Figure creation and reuse

Use these guides to create, export, and reuse Figure Composer figures. For complete
figure examples, use the {ref}`plotting gallery <how-to-plotting>`.

(how-to-gui-figure-composer)=

## Creating and exporting a reusable figure

1. Open the source data in ImageTool Manager.
2. Right-click the required ImageTool rows and choose {guilabel}`Add to Figure…`.
3. Set {guilabel}`Action` to {guilabel}`New Figure` and create the figure.
4. In Figure Composer, arrange the axes in {guilabel}`Layout` and add or edit the
   plotting operations in {guilabel}`Recipe`.
5. Check the rendered figure after each recipe change.

(how-to-gui-export-figure)=

### Exporting the figure

Set the output DPI, transparency, bounding box, and padding in the {guilabel}`Export`
tab. Export the figure and inspect the saved file at its final display size.

(how-to-gui-preserve-figure-recipe)=

### Preserving the figure recipe

Save the Manager workspace to preserve the figure, recipe, and sources together. Use
{guilabel}`Copy Code` when the source data already exists as Python variables. Use
{guilabel}`Copy Full Code` in the Manager when the copied code must also recreate the
recorded analysis inputs and operations.

Use {ref}`how-to-gui-reuse-figure-recipe` when the same layout and plotting operations
must be applied to another dataset. Use {ref}`how-to-plotting-figure-styles` for a
custom Matplotlib stylesheet.

(how-to-gui-start-empty-figure)=

## Creating a figure before all source data is available

1. In ImageTool Manager, choose {menuselection}`File --> New Empty Figure`.
2. In Figure Composer, use {guilabel}`Add…` in the {guilabel}`Sources` tab to select
   ImageTool rows. You can also drag rows from the Manager into Figure Composer.
3. Configure the axes in {guilabel}`Layout`.
4. Add recipe steps and assign each data-dependent step to the required source.
5. Check the rendered figure after you add or replace a source.

Use this procedure when you must define the layout or plotting recipe before all source
data is available. Save the Manager workspace to preserve the figure and its sources.

(how-to-gui-reuse-figure-recipe)=

(how-to-gui-replace-figure-sources)=

## Reusing a figure recipe with other data

1. Open the existing Figure Composer figure and the replacement data in the same
   Manager.
2. Right-click the replacement ImageTool row and choose {guilabel}`Add to Figure…`.
3. Select the target figure.
4. Set {guilabel}`Action` to {guilabel}`Replace Source`.
5. Select the source to replace and apply the change.
6. Inspect all axes for changed coordinate ranges, normalization, labels, and missing
   recipe inputs.

Use {guilabel}`Add New Step` when the data must be added as another plot. Use
{guilabel}`Add Source Only` when the recipe will refer to the source in a later manual
step.

Save the workspace under a new name when the original recipe and sources must remain
available.

(how-to-gui-install-figure-style)=

## Installing a custom stylesheet

1. Open the shared Settings window.
2. Select {guilabel}`Figure Composer`.
3. In {guilabel}`Stylesheets`, choose {guilabel}`Open Folder`.
4. Copy the `*.mplstyle` file into that folder.
5. Choose {guilabel}`Reload`.
6. Select the stylesheet name without the `.mplstyle` suffix, then choose
   {guilabel}`Add`.
7. Open or refresh a figure. Check its fonts, line widths, dimensions, and export
   settings.

If the style is not available, confirm that the file is in the folder opened by
Settings and that Matplotlib can parse it. See {ref}`figure-composer-styles` for style
and export precedence.
