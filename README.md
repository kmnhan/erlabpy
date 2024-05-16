<h1 align="center">
ERLabPy
</h1>
<p align="center">
    <a href="https://pypi.org/project/erlab/" alt="Supported Python Versions">
        <img src="https://img.shields.io/pypi/pyversions/erlab?logo=python&logoColor=white" /></a>
    <a href="https://pypi.org/project/erlab/" alt="PyPi">
        <img src="https://img.shields.io/pypi/v/erlab?logo=pypi&logoColor=white" /></a>
    <a href="https://anaconda.org/conda-forge/erlab" alt="Conda Version">
        <img src="https://img.shields.io/conda/vn/conda-forge/erlab?logo=condaforge&logoColor=white" /></a>
</p><p align="center">
    <a href="https://github.com/kmnhan/erlabpy/actions/workflows/release.yml" alt="Workflow Status">
        <img src="https://img.shields.io/github/actions/workflow/status/kmnhan/erlabpy/release.yml?logo=github&label=tests" /></a>
    <a href="https://erlabpy.readthedocs.io/" alt="Documentation Status">
        <img src="https://img.shields.io/readthedocs/erlabpy?logo=readthedocs&logoColor=white" /></a>
    <a href="https://results.pre-commit.ci/latest/github/kmnhan/erlabpy/main" alt="pre-commit.ci status">
        <img src="https://results.pre-commit.ci/badge/github/kmnhan/erlabpy/main.svg" /></a>
    <a href="https://codecov.io/gh/kmnhan/erlabpy" alt="Codecov Coverage">
        <img src="https://img.shields.io/codecov/c/github/kmnhan/erlabpy?logo=codecov&logoColor=white" /></a>
    <a href="https://github.com/astral-sh/ruff" alt="Ruff">
        <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" /></a>
</p>

A library that provides a set of tools and utilities to handle, manipulate, and
visualize data from condensed matter physics experiments, with a focus on angle-resolved
photoemission spectroscopy (ARPES).

*ERLabPy* is built on top of the popular scientific computing libraries
[*numpy*](https://numpy.org/), [*scipy*](https://scipy.org/), and
[*xarray*](https://xarray.pydata.org/), and is designed to be easy to use and integrate
with existing scientific Python workflows so that you can quickly get started with your
data analysis. The package is still under development, so if you have any questions or
suggestions, please feel free to contact us. We hope you find ERLabPy useful for your
research!

*ERLabPy* is developed and maintained by the electronic structure research
laboratory at Korea Advanced Institute of Science and Technology (KAIST).

## Features

- **Data Loading**: A flexible and extensible data loading system is included, capable
  of handling various data formats.
- **Data Manipulation**: A set of tools for manipulating and transforming data,
  including interpolation, masking and symmetrization is provided.
- **Plotting**: ERLabPy provides many different plotting functions for visualizing image
  data, including 2D and 3D plots. Publication-quality plots can be generated with
  minimal effort.
- **Fitting**: Several functions and models are implemented for fitting various types of
  data, including broadened Fermi-Dirac distributions, momentum distribution curves
  (MDCs), and energy distribution curves (EDCs).
- **Interactive Data Visualization**: ERLabPy includes many interactive plotting
  routines that resemble panels in Igor Pro that are very responsive and easy to use.
  See screenshots below.
- **Active Development**: ERLabPy is actively developed and maintained, with new
  features and improvements being added regularly.

## Screenshots

Most interactive windows support dark mode. Viewing this page from a supported browser
with dark mode enabled will show the dark mode screenshots.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/kmnhan/erlabpy/blob/main/docs/source/images/imagetool_dark.png?raw=true">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/kmnhan/erlabpy/blob/main/docs/source/images/imagetool_light.png?raw=true">
  <img alt="Imagetool in action." src="https://github.com/kmnhan/erlabpy/blob/main/docs/source/images/imagetool_light.png?raw=true">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/kmnhan/erlabpy/blob/main/docs/source/images/ktool_1_dark.png?raw=true">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/kmnhan/erlabpy/blob/main/docs/source/images/ktool_1_light.png?raw=true">
  <img alt="Imagetool in action." src="https://github.com/kmnhan/erlabpy/blob/main/docs/source/images/ktool_1_light.png?raw=true">
</picture>


## Getting Started

To get started, see [installation instructions](https://erlabpy.readthedocs.io/en/latest/getting-started.html).

## Documentation

The full documentation for ERLabPy is available on [Read the Docs](https://erlabpy.readthedocs.io/).

## Contributing

ERLabPy is an open-source project and we welcome contributions from the community. If
you find any bugs, issues, or have any suggestions, please open an issue
[here](https://github.com/kmnhan/erlabpy/issues).

If you have any questions or need help with using ERLabPy, please feel free to ask on
the [Discussions page](https://github.com/kmnhan/erlabpy/discussions).

If you would like to add a new feature or fix a bug yourself, we would love to have your
contribution. Feel free to fork the repository and submit a pull request with your
changes.

For more information on contributing, see our [Contributing page](https://erlabpy.readthedocs.io/en/latest/contributing.html).

## License

This project is licensed under the terms of the [GPL-3.0 License](LICENSE).
