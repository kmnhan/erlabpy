# Getting Started

Welcome to ERLabPy! This documentation will guide you through the installation process and provide an overview of the package.

If you are new to programming with Python, check out [Scientific Python Lectures](https://github.com/jrjohansson/scientific-python-lectures) as a great starting point.

Data structures in ERLabPy are represented using [xarray](https://docs.xarray.dev/){cite:p}`hoyer2017xarray`, which provides a powerful data structure for working with multi-dimensional arrays. Be sure to review the [xarray tutorial](https://tutorial.xarray.dev/) and the [xarray user guide](https://docs.xarray.dev/en/stable/index.html) to get familiar with it.

## Installing

:::{note}
Parts of this section are based on [Scipy’s installation guide](https://www.scipy.org/install/) and [NumPy’s installation guide](https://numpy.org/install/).
:::

The recommended method of installation depends on your preferred workflow. The common workflows can roughly be broken down into the following categories:

- **Project-based** (e.g. ``uv``, ``pixi``) *(recommended)*
- **Environment-based** (e.g. ``pip``, ``conda``) *(the traditional workflow)*
- **From source** *(for debugging and development)*

In project-based workflows, a project is a directory containing a manifest file describing the project, a lock-file describing the exact dependencies of the project, and the project’s (potentially multiple) environments.

In contrast, in environment-based workflows you install packages into an environment, which you can activate and deactivate from any directory. These workflows are well-established, but lack some reproducibility benefits of project-based workflows.

Choose the method that best suits your needs. If you’re unsure, start with the project-based workflow using ``uv``.

:::::{tab-set}
::::{tab-item} Project Based

### Installing with ``uv``

Here is a step-by-step guide to setting up a project to use ``erlab``, with ``uv``, a Python package manager.

1. Install uv following the instructions in the [uv documentation](https://docs.astral.sh/uv/getting-started/installation/).

2. Create a new project in a new subdirectory by executing the following in a terminal:

   ```bash
   uv init my-project
   cd my-project
   ```

   :::{hint}
   The second command changes directory into your project’s folder. You can name the project whatever you like.
   :::

3. Add the ``erlab`` package to your project with all recommended optional dependencies:

   ```bash
   uv add "erlab[complete]"
   ```

   :::{note}
   This will automatically install Python if you don’t already have it installed!
   :::

   :::{hint}
   You can also add other packages to your project in the same way, e.g. ``uv add matplotlib``.
   :::

::::

::::{tab-item} Environment Based

The two main tools that install Python packages are ``pip`` and ``conda``. Their functionality partially overlaps (both can install Python packages), but they also have differences:

- **Conda** is cross-language and can install Python along with non-Python libraries and tools (e.g. compilers, CUDA, HDF5).
- **Pip** installs packages from the Python Packaging Index (PyPI) for a particular Python install.
- **Conda** provides an integrated solution for managing packages, dependencies and environments, whereas with **pip** you might need additional tools.

### Installing with ``conda``

[Miniforge](https://conda-forge.org/download/) is the recommended way to install ``conda`` and ``mamba``. If you are new to conda, the Scikit-HEP project has a [great guide](https://scikit-hep.org/user/installing-conda) to get you started.

After creating an environment, install ``erlab`` from conda-forge as follows:

```bash
conda install -c conda-forge erlab
```

Or with the recommended dependencies:

```bash
conda install -c conda-forge erlab pyside6 hvplot ipywidgets
```

If you require other [optional dependencies](#optional-dependencies), append them to the above command.

:::{hint}
If you are using conda on macOS, you might experience degraded performance with the default BLAS and LAPACK libraries.

For Apple Silicon Macs, use [Accelerate](https://developer.apple.com/accelerate/):

```bash
conda install "libblas=*=*accelerate"
```

For Intel Macs, use [MKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html):

```bash
conda install "libblas=*=*mkl"
```

To prevent conda from switching back to the default libraries upon updating, see the [conda-forge documentation](https://conda-forge.org/docs/maintainer/knowledge_base/#switching-blas-implementation).
:::

### Installing with ``pip``

1. [Install Python](https://www.python.org/downloads/).

2. Create and activate a virtual environment with ``venv``.

   :::{hint}
   See [the tutorial in the Python Packaging User Guide](https://packaging.python.org/en/latest/tutorials/installing-packages/#creating-virtual-environments).
   :::

3. Install ``erlab`` with pip, including all recommended optional dependencies:

   ```bash
   python -m pip install erlab[complete] pyqt6
   ```

For a list of all available optional dependencies, see the [optional dependencies](#optional-dependencies) section.

::::

::::{tab-item} From Source

For advanced users and developers who want to customize, debug, or contribute to ERLabPy, see the [contributing](./contributing) guide for more information.

::::
:::::

## Importing

Once installed, you can import ERLabPy in your Python scripts or interactive sessions.

The recommended import conventions are:

```python
import erlab
import erlab.analysis as era
import erlab.interactive as eri
import erlab.plotting as eplt
```

A more comprehensive set of imports might look like:

```python
import erlab
import erlab.analysis as era
import erlab.interactive as eri
import erlab.plotting as eplt
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

xr.set_options(keep_attrs=True)
```

:::{note}
The interactive plotting module, `erlab.interactive`, requires a Qt library (such as PyQt6 or PySide6). If one is not installed, ERLabPy will notify you upon import.
:::

## Dependencies

ERLabPy depends on several scientific Python libraries. The table below lists some of the key packages and their roles in the scientific Python ecosystem:

| Package | Used in |
| ------- | ------- |
| [numpy](https://numpy.org/doc/stable/) | Computation and array manipulation, linear algebra |
| [scipy](https://docs.scipy.org/doc/scipy/index.html) | Linear algebra, signal processing, and image processing |
| [xarray](https://docs.xarray.dev/) | Data storage and manipulation |
| [numba](https://numba.pydata.org/) | Just-in-time compilation for significant speedups |
| [matplotlib](https://matplotlib.org) | Plotting |
| [lmfit](https://lmfit.github.io/lmfit-py/) | Solving optimization problems and curve fitting |

For interactive plotting, a Qt library such as PyQt6 or PySide6 is required. ERLabPy imports Qt bindings from [qtpy](https://github.com/spyder-ide/qtpy), which automatically selects the appropriate library based on what is installed.

See the [user-guide](./user-guide/index) to get started with ERLabPy!

## Optional dependencies

For a full list of dependencies and optional dependency groups, check the `[project]` and `[project.optional-dependencies]` sections in [pyproject.toml](https://github.com/kmnhan/erlabpy/blob/main/pyproject.toml):

```{literalinclude} ../../pyproject.toml
:language: toml
:start-at: dependencies = [
:end-before: [dependency-groups]
```

## Notes on compatibility

- ERLabPy supports Python 3.11 and later.
- There are some [known compatibility issues](https://github.com/kmnhan/erlabpy/issues/17) with PyQt5 and PySide2. It is recommended to use PyQt6 or PySide6 if possible.
