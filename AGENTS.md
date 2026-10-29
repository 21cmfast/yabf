# Notes for coding agents

See `CONTRIBUTING.rst` for setup, branches (`main` is v3, `v2.x` takes 2.x fixes) and
the checks to run before pushing (`ruff` and `pytest`).

## How we work

### Documentation: scope and audience

- **Docstrings are timeless and written for users.** Describe what the code does and
  how to use it. No internal jargon, no milestone or task names, no references to
  past issues, pull requests or earlier behaviour.
- **The README is short and to the point.**
- **User-facing documentation never refers to internal decisions** or how we got here.
- **Code comments may explain an internal decision** where that helps the next
  reader, linking the issue if there is one. Never use jargon that only you (the
  agent) know from the current session.

### Tests

- **Prefer physically or statistically meaningful tests** to "functionality" tests that
  only raise coverage. Before writing tests, think about which general physical or
  statistical conditions the code must satisfy (limits, symmetries, conservation,
  known analytic cases, calibration of posteriors and estimators), and test those.
- **Use `pytest.mark.parametrize`** instead of repeating near-identical tests.
- **No tests that exist only for coverage.** A test that a method we just defined
  exists, or that a constructor stores its arguments, tests nothing.

### Code

- **Look for an existing implementation before writing your own**, here and in our
  dependencies. When scoping a change, ask whether refactoring existing code would
  make it simpler.
- **Give every function and method type hints and a docstring**, private helpers
  included. One line is enough for a simple one. Small built-ins are what people
  copy when writing their own, so they set the standard.
- **Prefer immutable objects with clear transformations**, built with `attrs`
  (frozen classes; `evolve`/methods that return new objects rather than mutating).
- **Prefer composition over inheritance**, unless inheritance is exactly right for
  the job.
- **Keep backwards compatibility** unless the change is for a new major version. If
  keeping it means jumping through many hoops, stop and reconsider.
- **If following these rules makes the code needlessly complicated or ugly, stop and
  ask a human.**

### Before committing

- **Run the linters through the pre-commit hooks** (`uv run prek run --all-files`, or
  `uv run prek install` so they run on every commit) and fix what they report.
- **Don't silence a ruff rule on a line** (`# noqa`) or in the config without a very
  good reason; usually, check with a human first.

## Label every pull request

Always add labels when you open a pull request. Release Drafter builds the release notes
and the next version number from them, so an unlabelled PR is missing from the notes and
cannot bump the version. The labels are defined in `.github/labels.yml`. Use only those;
to add a new label, add it to that file in the same PR.

- **Exactly one `type:` label**: `type: bug`, `type: feature: ui`, `type: accuracy`,
  `type: testing`, `type: ci`, `type: performance: memory` / `cpu`, or a
  `type: maint: ...` label (`documentation`, `refactoring`, `removal`, `style`,
  `dependencies`, `build`). It picks the section in the release notes.
- **`API Breaking`** whenever the PR changes the public API in a way that breaks
  existing code. It makes the next release a major version.
  `type: feature: ui` makes it a minor version; anything else is a patch.
- **`context:` labels** for the parts of yabf the PR touches (`framework`, `mcmc`,
  `io`, `error-handling`, `hpc`), where one fits.
- **`downstream: 21cmMC`** when the change exists for, or affects, 21cmMC v2.

`status:`, `priority:` and `effort:` labels are for triaging issues; leave them off PRs.
