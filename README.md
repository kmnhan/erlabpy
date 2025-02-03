# ERLabPy

[![Supported Python Versions](https://img.shields.io/pypi/pyversions/erlab?logo=python&logoColor=white)](https://pypi.org/project/erlab/)
[![PyPi](https://img.shields.io/pypi/v/erlab?logo=pypi&logoColor=white)](https://pypi.org/project/erlab/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/erlab?logo=condaforge&logoColor=white)](https://anaconda.org/conda-forge/erlab)
[![Workflow Status](https://img.shields.io/github/actions/workflow/status/kmnhan/erlabpy/ci.yml?logo=github&label=tests)](https://github.com/kmnhan/erlabpy/actions/workflows/ci.yml)
[![Documentation Status](https://img.shields.io/readthedocs/erlabpy?logo=readthedocs&logoColor=white)](https://erlabpy.readthedocs.io/)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/kmnhan/erlabpy/main.svg)](https://results.pre-commit.ci/latest/github/kmnhan/erlabpy/main)
[![Codecov Coverage](https://img.shields.io/codecov/c/github/kmnhan/erlabpy?logo=codecov&logoColor=white)](https://codecov.io/gh/kmnhan/erlabpy)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![SPEC 1 — Lazy Loading of Submodules and Functions](https://img.shields.io/badge/SPEC-1-green?labelColor=%23004811&color=%235CA038)](https://scientific-python.org/specs/spec-0001/)

ERLabPy provides tools to handle, manipulate, and visualize data from condensed matter physics experiments, focusing on angle-resolved photoemission spectroscopy (ARPES).

ERLabPy integrates with scientific Python workflows for efficient data analysis.

## Features

- **Data Loading**: Flexible system for various data formats.
- **Data Manipulation**: Tools for interpolation, masking, and symmetrization.
- **Plotting**: Functions for 2D and 3D publication-quality plots.
- **Fitting**: Functions for fitting data, including Fermi-Dirac distributions, MDCs, EDCs, and more.
- **Interactive Visualization**: Responsive plotting routines similar to Igor Pro.

## Gallery

Interactive windows support dark mode.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/kmnhan/erlabpy/blob/main/docs/source/images/imagetool_dark.png?raw=true">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/kmnhan/erlabpy/blob/main/docs/source/images/imagetool_light.png?raw=true">
  <img alt="ImageTool in action." src="https://github.com/kmnhan/erlabpy/blob/main/docs/source/images/imagetool_light.png?raw=true">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/kmnhan/erlabpy/blob/main/docs/source/images/ktool_1_dark.png?raw=true">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/kmnhan/erlabpy/blob/main/docs/source/images/ktool_1_light.png?raw=true">
  <img alt="Interactive momentum conversion tool." src="https://github.com/kmnhan/erlabpy/blob/main/docs/source/images/ktool_1_light.png?raw=true">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/kmnhan/erlabpy/blob/main/docs/source/images/manager_dark.png?raw=true">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/kmnhan/erlabpy/blob/main/docs/source/images/manager_light.png?raw=true">
  <img alt="ImageTool manager window." src="https://github.com/kmnhan/erlabpy/blob/main/docs/source/images/manager_light.png?raw=true">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/kmnhan/erlabpy/blob/main/docs/source/images/explorer_dark.png?raw=true">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/kmnhan/erlabpy/blob/main/docs/source/images/explorer_light.png?raw=true">
  <img alt="Data explorer window." src="https://github.com/kmnhan/erlabpy/blob/main/docs/source/images/explorer_light.png?raw=true">
</picture>

## Getting Started

See [installation instructions](https://erlabpy.readthedocs.io/en/stable/getting-started.html).

## Documentation

Full documentation is available on [Read the Docs](https://erlabpy.readthedocs.io/).

## Contributing

We welcome contributions. Report issues [here](https://github.com/kmnhan/erlabpy/issues). For questions, visit the [Discussions page](https://github.com/kmnhan/erlabpy/discussions). To contribute, fork the repository and submit a pull request. See our [Contributing page](https://erlabpy.readthedocs.io/en/stable/contributing.html) for more information.

## License

Licensed under the [GPL-3.0 License](LICENSE).
