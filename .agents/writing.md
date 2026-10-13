# Agent instructions for commit and changelog text

Use the user's edited messages as the primary style reference. Consult recent entries
in `CHANGELOG.md` when drafting new messages, while preserving the user's current
preferences when older entries differ.

## Titles and tone

Use scoped Conventional Commits and include relevant issue or PR references when
available. Keep titles short and name the capability or corrected behavior that users
need to identify. Use established product names and scientific terms, with backticks
for exact identifiers.

Write for users of the feature. Use direct, connected sentences and prefer present
tense. Include technical details when they explain a scientific meaning, an input users
must supply, or a visible behavior. Put routine implementation and validation details
in the PR rather than the changelog description.

## Descriptions

- A small feature usually needs one sentence stating its capability and when it applies.
  Prefer descriptive openings with `Adds` or `Implements` rather than an instruction.
- For a larger feature, state its purpose first, then describe the main inputs,
  capabilities, and workflow in a short paragraph. Use a short list for independent
  controls or capabilities. Give enough detail for users to understand what they can
  do with it.
- For a fix, describe the failed user behavior and the condition that caused it.
  Prefer the opening `Fixes an issue where` when it fits.
- For an API or metadata rename, explain why the old name was misleading and identify
  the new name and meaning. Include compatibility details when users need them.
- A body is optional when the title explains the change. Straightforward documentation
  commits can have no body. Keep useful context rather than enforce a fixed length or
  sentence template.

## Commit types and delivery

Commit types select what appears in the changelog. Honor the user's classifications.
Use `chore` for implementation support or changes the user wants outside release notes,
including rarely used capabilities when requested. A metadata naming change can use
`refactor`. Use `feat` and `fix` for the capabilities and corrected behavior selected
for release notes.

Each commit still includes its relevant implementation, tests, and documentation.
Check claims against the current source. When the user requests an editable commit
series, return all titles and bodies in one plain text field, separated by blank lines,
without scope notes.
