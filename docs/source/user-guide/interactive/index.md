(interactive-tools)=

# Interactive tools

ERLab's interactive applications cover the complete workflow from exploratory slicing to managing large batches of ImageTool windows. The entries below highlight where to start depending on your background—whether you are coming from Igor Pro's original Image Tool or learning ARPES analysis from scratch in Python.

- [ImageTool](imagetool.md) is the primary GUI for inspecting multidimensional {class}`xarray.DataArray` objects. It includes cursor math, colormap controls, destructive and non-destructive edits, and hooks to ARPES-specific tools.
- [ImageTool manager](manager.md) sits on top of ImageTool and provides long-running project management: load queues, window linking, archiving, notebook synchronization, and workspace persistence.
- Specialized helpers such as {ref}`ktool <guide-ktool>` (momentum conversion), {ref}`dtool <guide-dtool>` (derivative plotting), and the {ref}`data explorer <guide-restool>` integrate with both ImageTool and the manager for targeted tasks.

Additional interactive tools, including conversion utilities and experiment planning aids, are described in {ref}`interactive-misc-tools`. Each tool can be launched directly or from ImageTool's context menus, and the manager can keep them alongside ImageTool windows when you need a shared workspace.

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
```
