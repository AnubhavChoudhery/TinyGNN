# =============================================================================
#  TinyGNN — Sphinx Documentation Configuration
#  docs/conf.py
#
#  Generates Python API documentation with C++ cross-references via Breathe.
#
#  Usage:
#    cd docs && make html
#
#  Output: docs/_build/html/index.html
# =============================================================================

import os
import sys

# -- Path setup ---------------------------------------------------------------
# Add project root so autodoc can import the Python package
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../python"))

# -- Project information ------------------------------------------------------
project = "TinyGNN"
copyright = "2025-2026, Jai Ansh Singh Bindra and Anubhav Choudhery (under JBAC EdTech)"
author = "Jai Ansh Singh Bindra and Anubhav Choudhery"
version = "0.1.0"
release = "0.1.0"

# -- General configuration ----------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "breathe",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Breathe configuration (Doxygen C++ → Sphinx) ----------------------------
breathe_projects = {
    "TinyGNN": os.path.abspath("../docs/doxygen/xml"),
}
breathe_default_project = "TinyGNN"
breathe_default_members = ("members", "undoc-members")

# -- Options for HTML output ---------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_title = "TinyGNN Documentation"
html_short_title = "TinyGNN"
html_show_sourcelink = True
html_show_copyright = True

html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "includehidden": True,
    "titles_only": False,
    "logo_only": False,
    "display_version": True,
}

# -- Intersphinx mapping (link to NumPy, SciPy, etc.) -------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# -- Napoleon settings (Google/NumPy docstring support) ------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True

# -- autodoc settings ----------------------------------------------------------
autodoc_member_order = "bysource"
autodoc_typehints = "description"
