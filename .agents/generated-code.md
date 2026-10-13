# Agent instructions for generated code

Use public APIs and semantic names in all user-facing generated, copied, and replay
code. Copied code is notebook-facing Python. Use direct public expressions, complete
assignments, and semantic names from the user's variables, provenance inputs, or visible
actions. Inline one-use aliases when safe. Avoid relay assignments, generic scratch
names reused across unrelated expressions, and private Manager helpers. Omit routine
framework imports for one operation; import each required module once in a composed
workflow.

Copied code represents the public operation. If that operation updates input metadata,
show the update directly. Keep exact source isolation and non-mutation in structured
replay and execution, rather than adding defensive copies or private restoration
machinery to copied code.

Test generated code with `exec()` in an explicit namespace and assert the exact result.
Check strings or formatting only when that text is the behavior under test. Cleanliness
checks must also execute the code and verify its result. Review emitted code for
redundant assignments and internal details; use property-style checks when cleanliness
is the behavior being tested.
