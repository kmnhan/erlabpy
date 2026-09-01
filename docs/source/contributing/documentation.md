(documentation)=

# Documentation contributions

ERLabPy documentation uses MyST Markdown and [Sphinx](https://www.sphinx-doc.org/).
For advanced documentation changes, see the [Sphinx documentation](https://www.sphinx-doc.org/en/master/).

The documentation has two parts:

- Source docstrings.
- Hand-written pages in `docs/source/`.

Docstrings and Reference pages describe individual functions and applications. The
remaining pages follow the documentation types below.

## Documentation types

ERLabPy documentation follows [Diátaxis](https://diataxis.fr/). Decide which user need
the page serves before you add content:

- A **tutorial** is a controlled and linear lesson. It uses fixed inputs, shows expected
  results, and contains only the explanation needed to complete the lesson.
- An **explanation** describes an ERLabPy design choice or scientific workflow that a
  user must understand. It can assume the knowledge taught in the tutorial. Do not use
  it to teach basic xarray concepts again.
- A **how-to guide** helps a competent user complete one real task. It must address a
  user goal, apply to the user's own work, provide an executable sequence, and contain
  only the decisions, checks, and recovery guidance needed for that task.
- A **reference** page describes the API or application accurately and completely. Its
  structure should follow the product that it describes.

Python API Reference is generated from public docstrings. Update the source docstring
when Python API Reference is incomplete or incorrect. Do not create a hand-written page
that duplicates function or class details. Use hand-written Reference for GUI
applications, installation, and other product lookup information.

Teach the required xarray concepts and ERLabPy data conventions in the controlled
tutorial path. Keep Explanation pages short. Prefer a table, list, diagram, or focused
figure when it communicates the relationship more clearly than prose. Put a figure that
demonstrates a procedure in the relevant How-to guide.

Tutorial code is the canonical example for a shared workflow. A How-to guide can reuse
that code for a concrete task. Link to the tutorial and keep the public API calls and
scientific conventions consistent. Do not add a competing version of the same example.

A feature does not belong in a tutorial or Explanation only because it is new. Add a
task procedure to How-to Guides and a GUI control to Reference. Add Explanation only
when the feature changes an ERLabPy design choice or a scientific workflow that users
must understand. Update a tutorial when the learning path itself changes. Do not move
option catalogs into a How-to guide because they do not fit the tutorial.

## Docstrings

Docstrings follow the [NumPy docstring standard](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard).
It specifies the format of each docstring section. See the [Sphinx NumPy-style examples](https://www.sphinx-doc.org/en/master/usage/extensions/example_numpy.html)
or similar existing functions for details.

Read the Docs updates the documentation when a commit is pushed to `main`.

Type annotations that follow {pep}`484` are recommended. Sphinx includes these
annotations in the documentation. You can omit type information from a docstring when
the function has complete annotations.

(building-the-documentation-locally)=

## Building the documentation locally

Clone the repository and change to its root directory. Make sure you have
{ref}`installed uv <installing-uv>`. Install the documentation dependencies:

```sh
uv sync --all-extras --dev --group docs
```

Build the HTML documentation:

```sh
uv run --directory docs make html
```

The HTML output is in `docs/build/html/`. Open `docs/build/html/index.html` in a local
browser to inspect the complete documentation site.

## Building the PDF locally

Read the Docs builds PDF documentation with LaTeX. To display emoji and special symbols
correctly, install these font families:

- `Noto Color Emoji`
- `Noto Sans Math`
- `IBM Plex Sans`
- `IBM Plex Mono`
- `DejaVu Sans Mono`

Build the PDF:

```sh
uv run --directory docs make latexpdf
```

The PDF output is `docs/build/latex/erlab.pdf`.

If the PDF has square replacement boxes, check the build log for missing glyphs:

```sh
rg "Missing character" docs/build/latex/erlab.log
```
