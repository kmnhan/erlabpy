# Contributing

Contributions can include bug reports, feature requests, code, tests, documentation,
data loaders, and interactive tools. Ask general questions in
[GitHub Discussions](https://github.com/kmnhan/erlabpy/discussions).

## Contribution paths

| Goal | Start here |
| --- | --- |
| Report a bug or request a feature | Use [GitHub Issues](https://github.com/kmnhan/erlabpy/issues). |
| Change code or tests | {doc}`contributing/development` covers environment setup, branches, local checks, pull requests, and code standards. |
| Improve the documentation | {doc}`contributing/documentation` covers documentation types, docstrings, and local builds. |
| Add support for an endstation | {doc}`contributing/loaders` covers loader implementation, metadata, multiple-file scans, summaries, and tests. |
| Add an interactive analysis tool | {doc}`contributing/interactive-tools` covers tool windows, Manager integration, state restoration, provenance, and tests. |

```{toctree}
:hidden: true
:maxdepth: 2

contributing/development
contributing/documentation
contributing/loaders
contributing/interactive-tools
```

## Bug reports and feature requests

Search the [issue tracker](https://github.com/kmnhan/erlabpy/issues) before you open an
issue. For a bug, include a minimal example, the observed result, the expected result,
and the relevant environment details. For guidance, see
[How to create a minimal reproducible example](https://stackoverflow.com/help/minimal-reproducible-example).

## Pull requests

Before you open a pull request:

- Keep the change focused on one feature or problem.
- Add or update tests for changed behavior.
- Run the checks that apply to the changed files.
- Build the documentation when you change documentation sources or public docstrings.
- Use a [Conventional Commit](https://www.conventionalcommits.org/en/v1.0.0/) title.
- Describe the behavior change and list the checks that you ran.

Use a draft pull request when you need feedback before the change is ready to merge.
See {doc}`contributing/development` for the complete development and submission
workflow.
