# conf.py

# -- Path setup --------------------------------------------------------------
# If your project's source code is in a different directory, you may need to
# add its path here so that Sphinx's autodoc extension can find it.
import os
import sys
# sys.path.insert(0, os.path.abspath('.'))
# sys.path.insert(0, os.path.abspath('../'))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Sahara Benchmark'
copyright = 'UBC DLNLP Lab'
author = 'Your Name'
release = '0.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Add any Sphinx extension module names here, as strings.
extensions = [
    # Allows Sphinx to automatically generate documentation from docstrings.
    'sphinx.ext.autodoc',
    # Allows Sphinx to understand Google-style and NumPy-style docstrings.
    'sphinx.ext.napoleon',
    # Adds links to the source code from the documentation.
    'sphinx.ext.viewcode',
    # Adds support for Markdown files (.md) in addition to reStructuredText (.rst).
    'myst_parser',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# The theme to use for HTML pages. The 'sphinx_rtd_theme' is a popular,
# mobile-friendly theme. You must install it separately.
# pip install sphinx_rtd_theme
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets or
# script files) here, relative to this directory. They are copied after
# the builtin static files, so a file named "default.css" will
# overwrite the builtin "default.css".
html_static_path = ['_static']

# -- MyST Parser Options -----------------------------------------------------
# https://myst-parser.readthedocs.io/en/latest/configuration.html

# Enable specific MyST extensions if you need them.
# myst_enable_extensions = [
#     "colon_fence",
# ]
