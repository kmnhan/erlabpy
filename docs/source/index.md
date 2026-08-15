# ERLabPy documentation

:::{only} format_html
**Date**: {sub-ref}`today`

```{image} https://img.shields.io/pypi/pyversions/erlab?style=flat-square&logo=python&logoColor=white
:alt: Supported Python Versions
:target: https://pypi.org/project/erlab/
```

```{image} https://img.shields.io/pypi/v/erlab?style=flat-square&logo=pypi&logoColor=white
:alt: PyPi
:target: https://pypi.org/project/erlab/
```

```{image} https://img.shields.io/conda/vn/conda-forge/erlab?style=flat-square&logo=condaforge&logoColor=white
:alt: Conda Version
:target: https://anaconda.org/conda-forge/erlab
```

```{image} https://img.shields.io/github/last-commit/kmnhan/erlabpy?style=flat-square&logo=github&color=lightseagreen
:alt: Last Commit
:target: https://github.com/kmnhan/erlabpy.git
```

:::

**ERLab**: **E**xtensible and **R**eproducible **L**ibrary for the **A**nalysis of **B**and structures

``erlab`` provides a complete Python workflow for ARPES (Angle-Resolved Photoemission
Spectroscopy) experiments. It provides a wide range of tools for processing, analyzing,
and visualizing ARPES data.

``erlab`` is built on top of the popular scientific Python libraries
[numpy](https://numpy.org), [scipy](https://scipy.org), and
[xarray](https://xarray.pydata.org), and is designed to be easy to use and integrate
with existing scientific Python workflows so that you can quickly get started with your
data analysis.

:::::{only} format_html
::::{grid} 1 1 2 2
:gutter: 1

:::{grid-item-card} Getting started
:link: getting-started
:link-type: doc
Install ERLabPy, verify the installation, and choose a learning path.
:::

:::{grid-item-card} Tutorials
:link: tutorials/index
:link-type: doc
Learn the Python or Manager workflow through guided lessons.
:::

:::{grid-item-card} Explanation
:link: explanation/index
:link-type: doc
Understand ERLabPy design choices and scientific workflows.
:::

:::{grid-item-card} How-to guides
:link: how-to/index
:link-type: doc
Complete a specific analysis, configuration, or GUI task.
:::

:::{grid-item-card} Reference
:link: reference
:link-type: doc
Look up Python APIs, GUI applications, settings, and supported formats.
:::

:::{grid-item-card} Contributing guide
:link: contributing
:link-type: doc
The contributing guide contains information on how to contribute to the project.
:::
::::
:::::

```{image} images/imagetool_light.png
:align: center
:alt: ImageTool window in light mode
:class: only-light
```

:::{only} format_html

```{image} images/imagetool_dark.png
:align: center
:alt: ImageTool window in dark mode
:class: only-dark
```

:::

```{toctree}
:caption: Contents
:hidden: true
:maxdepth: 3

getting-started
tutorials/index
explanation/index
how-to/index
reference
contributing
bibliography
changelog
```
