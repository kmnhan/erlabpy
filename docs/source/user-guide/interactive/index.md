(interactive-tools)=

# Interactive tools

ERLab's interactive applications cover the complete workflow from exploratory slicing to managing large batches of ImageTool windows and specialized tools. These applications are designed to be used alongside Jupyter notebooks, allowing you to combine interactive exploration with reproducible analysis.

If you are moving back and forth between notebook code and the GUI, start with
{ref}`workflow-bridge`. It maps the most common interactive workflows to the
corresponding public Python APIs and explains how to round-trip between them.

- [ImageTool](imagetool.md) is the primary GUI for inspecting multidimensional {class}`xarray.DataArray` objects. It includes navigation with multiple cursors, colormap controls, destructive and non-destructive edits, and quick access to analysis and plotting tools.
- [ImageTool manager](manager.md) sits on top of ImageTool and provides long-running project management: loader integration, cursor linking, notebook synchronization, and savable workspaces. It can also manage other interactive tools, keeping them organized and accessible.
- Specialized helpers such as {ref}`ktool <guide-ktool>` (momentum conversion), {ref}`dtool <guide-dtool>` (derivative plotting), and the {ref}`data explorer <guide-data-explorer>` integrate with both ImageTool and the manager for specific analysis tasks. These are described in {ref}`interactive-misc-tools`. Each tool can be launched directly or from the relevant ImageTool menu or context menu, and the manager can keep them alongside ImageTool windows when you need a shared workspace.
- Experienced users who want to build or contribute a new GUI should continue to
  {ref}`interactive-tool-authoring`.

:::{tip}

A recommended workflow is to perform data loading and analysis within a Jupyter notebook, while using the ImageTool manager to handle interactive visualization.

This approach lets you take advantage of ImageTool's interactive features while maintaining a reproducible analysis process in your notebook. For details on integrating ImageTool manager with Jupyter notebooks, see {ref}`working-with-notebooks`.

:::

```{toctree}
:caption: Table of Contents
:maxdepth: 2

imagetool
manager
misc-tools
options
tool-authoring
```
