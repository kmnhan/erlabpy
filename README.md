# ERLabPy

[![Workflow Status](https://github.com/kmnhan/erlabpy/actions/workflows/release.yml/badge.svg)](https://github.com/kmnhan/erlabpy/actions/workflows/release.yml)
[![Documentation Status](https://readthedocs.org/projects/erlabpy/badge/?version=latest)](https://erlabpy.readthedocs.io/en/latest/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Welcome to *ERLabPy*, a Python library designed to simplify and streamline the
process of analyzing data from condensed matter physics experiments.

This library provides a set of tools and utilities to handle, manipulate, and
visualize data, with a focus on data from angle-resolved photoemission
spectroscopy (ARPES) experiments.

*ERLabPy* is built on top of the popular scientific computing libraries `numpy`,
`scipy`, and `xarray`, and is designed to be easy to use and integrate with
existing scientific Python workflows. It is also designed to be extensible,
allowing users to easily add custom functionality and analysis tools.

*ERLabPy* is developed and maintained by the Electronic structure Research
Laboratory (ERLab) at Korea Advanced Institute of Science and Technology
(KAIST).

## Features

- **Data Loading**: A flexible and extensible data loading system is included,
  capable of handling various data formats.
- **Data Manipulation**: A set of tools for manipulating and
  transforming data, including interpolation, masking and symmetrization is provided.
- **Plotting**: ERLabPy provides many different plotting functions for visualizing
  image data, including 2D and 3D plots. Publication-quality plots can be
  generated with minimal effort.
- **Fitting**: Several functions and models are implemented for fitting various
  types of data, including broadened Fermi-Dirac distributions, momentum
  distribution curves (MDCs), and energy distribution curves (EDCs).
- **Interactive Data Visualization**: ERLabPy includes many interactive plotting
  routines that resemble panels in Igor Pro that are very responsive and easy to
  use. See screenshots below.
- **Active Development**: ERLabPy is actively developed and maintained, with new
  features and improvements being added regularly.

## Screenshots

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/kmnhan/erlabpy/blob/main/docs/source/images/imagetool_dark.png?raw=true">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/kmnhan/erlabpy/blob/main/docs/source/images/imagetool_light.png?raw=true">
  <img alt="Imagetool in action." src="https://github.com/kmnhan/erlabpy/blob/main/docs/source/images/imagetool_light.png?raw=true">
</picture>
Try enabling/disabling dark mode in your browser!

## Documentation

The full documentation for ERLabPy is available at [Read the Docs](https://erlabpy.readthedocs.io/en/latest/).

## Getting Started

To get started with ERLab, follow the [installation instructions](https://erlabpy.readthedocs.io/en/latest/readme.html).

Then, you can import the `erlab` module in your Python scripts and start analyzing your data.

```python
import erlab
```

## Contributing

Contributions are welcome! Please open an issue or pull request if you have any
suggestions or improvements.

## License

This project is licensed under the terms of the [GPL-3.0 License](LICENSE).

## Contact

If you have any questions, issues, or suggestions, please open an issue on this
repository. We appreciate your feedback!
