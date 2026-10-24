# Notes for coding agents

See `CONTRIBUTING.rst` for setup, branches (`main` is v3, `v2.x` takes 2.x fixes) and
the checks to run before pushing (`ruff` and `pytest`).

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
