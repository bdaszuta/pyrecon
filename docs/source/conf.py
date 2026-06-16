# Configuration file for the Sphinx documentation builder.
#
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import re
import sys

# -- Path setup ----------------------------------------------------------
# Point autodoc at the pyrecon package source (two levels up from docs/source/)
sys.path.insert(0, os.path.abspath(
  os.path.join(os.path.dirname(__file__), '..', '..')))

# -- Project information -------------------------------------------------
project = 'pyrecon'
copyright = '2026, Boris Daszuta'
author = 'Boris Daszuta'
release = '0.1.0'

# -- General configuration -----------------------------------------------
extensions = [
  'sphinx.ext.autodoc',
  'sphinx.ext.napoleon',
  'sphinx.ext.viewcode',
  'sphinx.ext.intersphinx',
]

templates_path = ['_templates']
exclude_patterns = []

language = 'en'

# Suppress RST parser warnings from module-level ASCII-art headers
# that autodoc feeds to docutils.  These are all benign.
suppress_warnings = ['docutils']

# -- Autodoc configuration -----------------------------------------------
autodoc_default_options = {
  'members': True,
  'undoc-members': False,
  'show-inheritance': False,
}
autodoc_typehints = 'none'

# -- Napoleon configuration (Google-style docstrings) --------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False

# -- Intersphinx ---------------------------------------------------------
intersphinx_mapping = {
  'python': ('https://docs.python.org/3', None),
  'numpy': ('https://numpy.org/doc/stable/', None),
}

# -- Options for HTML output ---------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_theme_options = {}
html_static_path = ['_static']
html_title = 'pyrecon'
html_short_title = 'pyrecon'


# -- Event handlers ------------------------------------------------------

_PYRECON_HEADER_RE = re.compile(
  r'\A\s*,-[*]\s*\n\s*[(]_[)]\s*\n(?:@author:[^\n]*\n)?'
  r'(?:@SPDX-License-Identifier:[^\n]*\n)?'
  r'(?:@function:\s*([^\n]*))?'
)

def _strip_pyrecon_header(app, what, name, obj, options, lines):
    """Strip ASCII-art banner + @author/@SPDX from module docstrings."""
    if what != 'module' or not lines:
        return
    text = '\n'.join(lines)
    m = _PYRECON_HEADER_RE.match(text)
    if m and m.group(1):
        lines[:] = [m.group(1)]


def setup(app):
    app.connect('autodoc-process-docstring', _strip_pyrecon_header)
