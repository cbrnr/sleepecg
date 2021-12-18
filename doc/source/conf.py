# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a
# full list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import inspect
import os
import sys

import sleepecg

# -- Project information --------------------------------------------------
project = 'SleepECG'
author = 'Florian Hofer'
copyright = '2021, SleepECG Developers'
version = sleepecg.__version__

# -- General configuration ------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.linkcode',
    'numpydoc',
    'myst_parser',
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

myst_enable_extensions = [
    'substitution',
]

myst_substitutions = {
    'version': version,
}

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/devdocs', None),
    'scipy': ('https://scipy.github.io/devdocs', None),
}

default_role = 'code'

autoclass_content = 'class'
autodoc_inherit_docstrings = False
autodoc_mock_imports = ['scipy', 'tqdm']
autodoc_typehints = 'none'
autosummary_generate = True
html_show_sourcelink = False

add_function_parentheses = False

numpydoc_class_members_toctree = False
numpydoc_show_class_members = False
numpydoc_show_inherited_class_members = False
numpydoc_xref_param_type = True

templates_path = ['_templates']

# -- Options for HTML output ----------------------------------------------
html_theme = 'pydata_sphinx_theme'
html_theme_options = {
    'github_url': 'https://github.com/cbrnr/sleepecg',
    'show_prev_next': False,
}


def linkcode_resolve(domain, info):
    """
    Determine the URL corresponding to a python object.

    Adapted from lasagne:
    https://github.com/Lasagne/Lasagne/blob/master/docs/conf.py
    """
    def find_source():
        obj = sys.modules[info['module']]
        for part in info['fullname'].split('.'):
            obj = getattr(obj, part)
        fn = inspect.getsourcefile(obj)
        fn = os.path.relpath(fn, start=os.path.dirname(sleepecg.__file__))
        source, lineno = inspect.getsourcelines(obj)
        return fn, lineno, lineno + len(source) - 1

    if domain != 'py' or not info['module']:
        return None
    try:
        filename = 'sleepecg/%s#L%d-L%d' % find_source()
    except Exception:
        filename = info['module'].replace('.', '/') + '.py'
    tag = 'main' if 'dev' in version else ('v' + version)
    return 'https://github.com/cbrnr/sleepecg/blob/%s/%s' % (tag, filename)
