# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a
# full list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import inspect
import os
import re
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
    'm2r2',
]

source_suffix = ['.rst', '.md']

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/devdocs', None),
    'scipy': ('https://scipy.github.io/devdocs', None),
}

autoclass_content = 'class'
autodoc_inherit_docstrings = False
autodoc_mock_imports = ['scipy', 'tqdm']
autodoc_typehints = 'none'
autosummary_generate = True
html_show_sourcelink = False

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
    Determine the URL corresponding to Python object.

    Adapted from SciPy (doc/source/conf.py).
    """
    if domain != 'py':
        return None

    modname = info['module']
    fullname = info['fullname']

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split('.'):
        try:
            obj = getattr(obj, part)
        except Exception:
            return None

    try:
        fn = inspect.getsourcefile(obj)
    except Exception:
        fn = None
    if not fn:
        try:
            fn = inspect.getsourcefile(sys.modules[obj.__module__])
        except Exception:
            fn = None
    if not fn:
        return None

    try:
        source, lineno = inspect.getsourcelines(obj)
    except Exception:
        lineno = None

    if lineno:
        linespec = '#L%d-L%d' % (lineno, lineno + len(source) - 1)
    else:
        linespec = ''

    startdir = os.path.abspath('../../sleepecg')
    fn = os.path.relpath(fn, start=startdir).replace(os.path.sep, '/')

    if fn.startswith('sleepecg/'):
        m = re.match(r'^.*dev0\+([a-f0-9]+)$', version)
        if m:
            return 'https://github.com/cbrnr/sleepecg/blob/%s/%s%s' % (
                m.group(1), fn, linespec,
            )
        elif 'dev' in version:
            return 'https://github.com/cbrnr/sleepecg/blob/main/%s%s' % (
                fn, linespec,
            )
        else:
            return 'https://github.com/cbrnr/sleepecg/blob/v%s/%s%s' % (
                version, fn, linespec,
            )
    else:
        return None
