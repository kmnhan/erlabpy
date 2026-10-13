# Agent instructions for documentation and API changes

Follow the existing [documentation types](../docs/source/contributing/documentation.md#documentation-types)
when choosing a page and its scope.

## Writing explanations

Build an explanation around a concrete user question or physical situation. Explain
the relevant relation or design choice, then connect it to the user's interpretation
or decision. Introduce equations after their purpose is clear, and define their symbols
and assumptions. Shortening text must preserve the reasoning that connects the facts.

Use connected paragraphs for causal reasoning, tables and lists for comparisons or
independent facts, and focused figures when they clarify a relationship. Include
implementation details and caveats only when they affect the interpretation or decision
being explained. Before finalizing, check that each paragraph helps the reader
understand why the behavior occurs or when it matters.

## Terminology and API changes

Read related pages to reuse their established terms and notation. Check equations
against the stated dimensions, units, and assumptions. Describe visible objects and
actions in concrete terms.

For public API behavior changes, add a `versionchanged` directive in the relevant
docstring or page. Verify the next release version; behavior changes will usually use
the next minor version. Use `versionchanged` and `versionadded` sparingly, when use on
older versions would be confusing or give ambiguous errors.

When moving or renaming pages, add redirects to `rediraffe_redirects` in
`docs/source/conf.py`.
When changing analysis guidance or pages linked by the repository's
[ARPES analysis skill](../skills/arpes-analysis/SKILL.md), check its guidance and affected
references for consistency. Update them when needed.
