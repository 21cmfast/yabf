"""Configuration for the Sphinx documentation."""

import yabf

project = "yabf"
author = "Steven Murray"
copyright = "2019, Steven Murray"
version = release = yabf.__version__

extensions = [
    "myst_nb",
    "numpydoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
]
master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
pygments_style = "sphinx"
html_theme = "alabaster"

# The tutorials are MyST notebooks, run on every build; an error in one fails it.
nb_execution_mode = "cache"
nb_execution_raise_on_error = True
nb_execution_timeout = 600

numpydoc_show_class_members = False
autodoc_member_order = "bysource"
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}

# Wide enough for code cells of 88 characters, the line length the code uses.
html_theme_options = {"page_width": "1200px", "body_max_width": "none"}
