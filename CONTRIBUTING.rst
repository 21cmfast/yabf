.. highlight:: shell

============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/21cmfast/yabf/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

yabf could always use more documentation, whether as part of the
official yabf docs, in docstrings, or even on the web in blog posts,
articles, and such.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at https://github.com/21cmfast/yabf/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

Where work goes
---------------

* ``main`` is the development branch for **yabf v3**, which changes the public API.
  New features and anything that breaks the v2 API go here.
* ``v2.x`` gets **bug fixes for yabf 2.x only**. Open those pull requests against
  ``v2.x``. A fix that also applies to v3 is ported to ``main`` as well.

Get Started!
------------

Ready to contribute? Here's how to set up ``yabf`` for local development.

1. Fork the ``yabf`` repo on GitHub (members of the 21cmFAST organisation can clone
   it directly).
2. Clone your fork locally::

    $ git clone git@github.com:your_name_here/yabf.git

3. Create an environment with the development dependencies and the pre-commit hooks,
   using `uv <https://docs.astral.sh/uv/>`_::

    $ cd yabf/
    $ uv sync --group dev
    $ uv run prek install

4. Create a branch for local development, from ``main`` (or from ``v2.x`` for a 2.x
   bug fix)::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

5. When you're done making changes, check that they pass ``ruff`` and the tests::

    $ uv run ruff check . && uv run ruff format --check .
    $ uv run pytest

6. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

7. Submit a pull request through the GitHub website, against ``main`` or ``v2.x``.

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring.
3. The pull request should pass CI on all supported Python versions (3.12+).

Tips
----

To run a subset of tests::

$ uv run pytest tests/test_params.py


Deploying
---------

A reminder for the maintainers on how to deploy. Release Drafter keeps a draft
release for ``main`` up to date; its version goes up by major for a pull request
labelled ``API Breaking``, minor for ``type: feature: ui``, else patch. Publish the
draft on GitHub; the ``deploy`` workflow then builds the package and uploads it to
PyPI. For a 2.x patch release, draft a release by hand with target ``v2.x`` and the
next 2.x tag (e.g. ``v2.1.5``).
