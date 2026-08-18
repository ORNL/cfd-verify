# Configuration file for the Sphinx documentation builder.

# -- Project information -----------------------------------------------------
project = "CFDverify"
copyright = "2025, Oak Ridge National Laboratory"
author = "Justin Weinmeister"
version = "0.0"
release = "0.0.2"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
]

autoclass_content = "both"

# HTML options
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_logo = "_static/cfdverify_wordmark.svg"
html_favicon = "_static/favicon.png"
html_css_files = ["custom.css"]
html_theme_options = {
    "logo_only": True,
    "style_nav_header_background": "#EAF3DE",
}
