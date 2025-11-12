# Interactive tools

This section introduces interactive tools designed for data visualization and manipulation.

The primary tool is [ImageTool](imagetool.md), a graphical user interface for exploring and interacting with multidimensional data. Its capabilities are further extended by the [ImageTool manager](manager.md), which streamlines the management of multiple ImageTool windows.

Additional interactive tools, such as those for momentum conversion, are described in [Miscellaneous Tools](misc-tools.md). These auxiliary tools can be launched as separate windows from within ImageTool and are also integrated into the ImageTool manager, enabling efficient organization and workflow.

:::{tip}

A recommended workflow is to perform data loading and analysis within a Jupyter notebook, while using the ImageTool manager to handle interactive visualization.

This approach lets you take advantage of ImageTool's interactive features while maintaining a reproducible analysis process in your notebook. For details on integrating ImageTool manager with Jupyter notebooks, see [](working-with-notebooks).

:::

```{toctree}
:caption: Table of Contents
:maxdepth: 2

imagetool
manager
misc-tools
options
```
