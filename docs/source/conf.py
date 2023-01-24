# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


import os
import sys
#sys.path.insert(0, os.path.abspath('../../src/lcdtreespace'))
sys.path.insert(0, os.path.abspath('../../src'))


project = 'lcdtreespace'
copyright = '2023, Yuki Takazawa'
author = 'Yuki Takazawa'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

#extensions = [
#    'sphinx.ext.duration',
#    'sphinx.ext.doctest',
#    'sphinx.ext.autodoc',
#    'sphinx.ext.autosummary',
#    'sphinx.ext.todo',
#    'sphinx.ext.viewcode',
#    'sphinx.ext.napoleon',
#]
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
    'sphinx.ext.napoleon',
    'sphinx.ext.githubpages'
]

templates_path = ['_templates']
exclude_patterns = []

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'
html_context = {
  'display_github': True,
  'github_user': 'yukiregista',
  'github_repo': 'LCDTreeSpace',
  'github_version': 'develop/docs/source/',
}

html_static_path = ['_static']

def skip(app, what, name, obj, would_skip, options):
    if name == "__init__":
        return False
    return would_skip

def setup(app):
    app.connect("autodoc-skip-member", skip)
