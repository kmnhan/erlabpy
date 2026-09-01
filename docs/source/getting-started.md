(getting-started)=

# Getting Started

ERLabPy provides two main workflows for ARPES analysis: a notebook-based Python workflow
and a GUI-centered workflow using [ImageTool Manager](imagetool-manager). Install
ERLabPy first, verify the installation, and then choose the learning path that matches
your work.

(installing)=

## Installing

Use Conda with [Miniforge](https://conda-forge.org/download/) if you are new to Python
package management. Select an installation method below. The additional packages in
each command enable the Qt applications and notebook widgets used in the tutorials.

:::::{tab-set}
::::{tab-item} Conda

Create or activate a Conda environment, then run:

```bash
conda install -c conda-forge erlab pyqt6 ipywidgets
```

[Miniforge](https://conda-forge.org/download/) is the recommended Conda distribution.
The [Scikit-HEP Conda guide](https://scikit-hep.org/user/installing-conda) explains how
to create and use environments.

:::{tip}

On macOS, the default BLAS and LAPACK libraries can reduce numerical performance.
On an Apple Silicon Mac, use Accelerate:

```bash
conda install "libblas=*=*_newaccelerate"
```

On an Intel Mac, use MKL:

```bash
conda install "libblas=*=*mkl"
```

See the [conda-forge BLAS documentation](https://conda-forge.org/docs/maintainer/knowledge_base/#switching-blas-implementation)
to keep this selection when you update the environment.
:::

::::

::::{tab-item} Pip

Create and activate a virtual environment, then run:

```bash
python -m pip install "erlab[complete]" pyqt6
```

See the [Python Packaging User Guide](https://packaging.python.org/en/latest/tutorials/installing-packages/)
if you need help creating a virtual environment.

::::

::::{tab-item} uv

Create a project and add ERLabPy:

```bash
uv init my-project
cd my-project
uv add "erlab[complete]" pyqt6
```

See the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/)
if `uv` is not installed.

::::

::::{tab-item} Pixi

Create a workspace and add ERLabPy:

```bash
pixi init my-project
cd my-project
pixi add erlab pyqt6 ipywidgets
```

See the [Pixi installation guide](https://pixi.prefix.dev/) if `pixi` is not installed.

::::

::::{tab-item} Standalone application

Download ImageTool Manager from [GitHub Releases](https://github.com/kmnhan/erlabpy/releases)
if you want to use the GUI without installing Python. Follow the
{ref}`platform instructions <imagetool-manager-standalone>` after downloading it.

::::
:::::

:::{important}

Install PyQt6 or PySide6 in the same environment as ERLabPy to use ImageTool and the
other interactive tools. The commands above install PyQt6. Replace `pyqt6` with
`pyside6` if you prefer PySide6. Qt5 bindings are not supported.
:::

For optional dependency groups, compatibility details, and platform notes, see
{doc}`reference/installation`. To install a development checkout, see
{ref}`creating-a-development-environment`.

(verify-installation)=

## Verifying the installation

Print the installed ERLabPy version:

```bash
python -c "import erlab; print(erlab.__version__)"
```

Confirm that Qt is available:

```bash
python -c "from qtpy import API_NAME; print(API_NAME)"
```

Launch ImageTool Manager:

```bash
itool-manager
```

## Before the tutorial

The Python tutorial assumes that you can run notebook cells, import packages, call
functions, and read basic Python errors. The tutorial introduces the xarray structure
and ERLabPy data conventions that it uses.

Use these resources if you need a review:

- [Scientific Python Lectures](https://lectures.scientific-python.org)
- [The Python tutorial](https://docs.python.org/3/tutorial/)
- [The xarray tutorial](https://tutorial.xarray.dev/)
- [The xarray user guide](https://docs.xarray.dev/en/stable/)

:::{tip}

For notebook work, use [Visual Studio Code](https://code.visualstudio.com) with the
[Jupyter extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)
and the [ERLab extension](https://marketplace.visualstudio.com/items?itemName=khan.erlab).
:::

## Choosing a workflow

{doc}`tutorials/python/index` is the recommended learning path. It uses one generated
dataset to introduce inspection, coordinate selection, plotting, and momentum
conversion. It also shows how to open the data in ImageTool from Python.

To understand how GUI and Python work together, see {ref}`workflow-bridge`. To learn the
current Manager workflow without a notebook, start the {ref}`manager-tutorial`.

The two workflows can exchange data and reproducible code. See {ref}`workflow-bridge`
for the relationship between them.

## AI assistance

- The `arpes-analysis` agent skill helps coding agents answer questions about ERLabPy
  and perform basic analysis tasks. With the [GitHub CLI](https://cli.github.com/)
  installed, run:

  ```bash
  gh skill install kmnhan/erlabpy arpes-analysis
  ```

- The [ARPES Analysis Assistant](https://chatgpt.com/g/g-6962fdab53008191ac5e3307a694b0a9-arpes-analysis-assistant)
  provides ChatGPT-based help. It can give code examples, explain functions and
  parameters, and give general guidance about installing and using ERLabPy.

:::{warning}

Large language models are not a source of truth. Verify API details in the Reference
section. Generated code can contain errors. Review and test it, and validate its
analysis on your data.
:::
