# ERLabPy
[![Supported Python Versions](https://img.shields.io/pypi/pyversions/erlab)](https://pypi.org/project/erlab/)
[![PyPi](https://img.shields.io/pypi/v/erlab.svg)](https://pypi.org/project/erlab/)
[![Workflow Status](https://github.com/kmnhan/erlabpy/actions/workflows/release.yml/badge.svg)](https://github.com/kmnhan/erlabpy/actions/workflows/release.yml)
[![Documentation Status](https://readthedocs.org/projects/erlabpy/badge/)](https://erlabpy.readthedocs.io/)
[![License](https://img.shields.io/pypi/l/erlab)](https://github.com/kmnhan/erlabpy/blob/main/LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/kmnhan/erlabpy/main.svg)](https://results.pre-commit.ci/latest/github/kmnhan/erlabpy/main)

A library that provides a set of tools and utilities to handle, manipulate, and
visualize data from condensed matter physics experiments, with a focus on
angle-resolved photoemission spectroscopy (ARPES).

*ERLabPy* is built on top of the popular scientific computing libraries `numpy`,
`scipy`, and `xarray`, and is designed to be easy to use and integrate with
existing scientific Python workflows. It is also designed to be extensible,
allowing users to easily add custom functionality and analysis tools.

*ERLabPy* is developed and maintained by the electronic structure research
laboratory at Korea Advanced Institute of Science and Technology (KAIST).

## Features

- **Data Loading**: A flexible and extensible data loading system is included,
  capable of handling various data formats.
- **Data Manipulation**: A set of tools for manipulating and transforming data,
  including interpolation, masking and symmetrization is provided.
- **Plotting**: ERLabPy provides many different plotting functions for
  visualizing image data, including 2D and 3D plots. Publication-quality plots
  can be generated with minimal effort.
- **Fitting**: Several functions and models are implemented for fitting various
  types of data, including broadened Fermi-Dirac distributions, momentum
  distribution curves (MDCs), and energy distribution curves (EDCs).
- **Interactive Data Visualization**: ERLabPy includes many interactive plotting
  routines that resemble panels in Igor Pro that are very responsive and easy to
  use. See screenshots below.
- **Active Development**: ERLabPy is actively developed and maintained, with new
  features and improvements being added regularly.

## Screenshots

Most interactive windows support dark mode. Viewing this page from a supported
browser with dark mode enabled will show the dark mode screenshots.

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


## Documentation

The full documentation for ERLabPy is available on [Read the Docs](https://erlabpy.readthedocs.io/).

## Getting Started

To get started with ERLab, follow the [installation instructions](https://erlabpy.readthedocs.io/en/stable/getting-started.html).

## Contributing

Contributions are welcome! Please open an issue or pull request if you have any
suggestions or improvements. For more information on contributing, see the
[development guide](https://erlabpy.readthedocs.io/en/stable/development.html).

## License

This project is licensed under the terms of the [GPL-3.0 License](LICENSE).

## Contact

If you have any questions, issues, or suggestions, please open an issue
[here](https://github.com/kmnhan/erlabpy/issues). We appreciate your feedback!
