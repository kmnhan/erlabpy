(dependencies-and-compatibility)=

# Dependencies and compatibility

ERLabPy supports Python 3.11 and later. Qt5 bindings are not supported. Interactive
tools require PyQt6 or PySide6 and import the selected binding through
[qtpy](https://github.com/spyder-ide/qtpy).

## Core dependencies

| Package | Used for |
| ------- | -------- |
| [numpy](https://numpy.org/doc/stable/) | Array operations and linear algebra |
| [scipy](https://docs.scipy.org/doc/scipy/index.html) | Signal, image, and numerical routines |
| [xarray](https://docs.xarray.dev/) | Labeled multidimensional data |
| [numba](https://numba.pydata.org/) | Just-in-time compilation |
| [matplotlib](https://matplotlib.org) | Static plotting |
| [lmfit](https://lmfit.github.io/lmfit-py/) | Optimization and curve fitting |

(optional-dependencies)=

## Optional dependencies

The package metadata is the source of truth for dependency versions and optional
groups:

```{literalinclude} ../../../pyproject.toml
:language: toml
:start-at: dependencies = [
:end-before: [project.urls]
```

```{literalinclude} ../../../pyproject.toml
:language: toml
:start-at: [project.optional-dependencies]
:end-before: [dependency-groups]
```
