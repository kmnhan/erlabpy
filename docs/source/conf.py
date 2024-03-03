# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "ERLabPy"
copyright = "2023, Kimoon Han"
author = "Kimoon Han"
release = "0.1"

# Documentation build dependencies
# sphinx, sphinx-autodoc-typehints, furo
# pypi: sphinx-qt-documentation


# build docs with PyQt6 since PySide6 is broken
# https://bugreports.qt.io/browse/PYSIDE-1884
import os

os.environ["QT_API"] = "pyqt6"


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    # "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "matplotlib.sphinxext.plot_directive",
    "matplotlib.sphinxext.figmpl_directive",
    # "IPython.sphinxext.ipython_directive",
    # "IPython.sphinxext.ipython_console_highlighting",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    #   'sphinx.ext.doctest',
    # "sphinx.ext.inheritance_diagram",
    "sphinx_qt_documentation",
    "nbsphinx",
]

plot_srcset = ["2x"]

templates_path = ["_templates"]
exclude_patterns = []


default_role = "obj"
add_function_parentheses = True
add_module_names = False
toc_object_entries = True
toc_object_entries_show_parents = "domain"

nitpicky = False
nitpick_ignore = [("py:class", "numpy.float64")]


autosummary_generate = True
autosummary_generate_overwrite = True

autodoc_inherit_docstrings = False
autodoc_typehints = "both"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    # "exclude-members":("sigDataChanged",),
    "show-inheritance": False,
}
# autodoc_typehints_description_target = "all"

# autodoc_type_aliases = {
# "numpy.float64": "float",
# "float64": "float",
# }

# Napoleon settings
napoleon_google_docstring = False
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
napoleon_preprocess_types = True
napoleon_type_aliases = {
    "array-like": "ndarray <numpy.ndarray>",
    "numpy.float64": "float",
    "float64": "float",
}
napoleon_attr_annotations = True
napoleon_custom_sections = [("Signals", "params_style")]
# napoleon_custom_sections = [("Signals", "Methods")]
# napoleon_custom_sections = [("Signals", "Attributes")]


qt_documentation = "PyQt6"
intersphinx_mapping = {
    "PyQt5": ("https://doc.qt.io/qtforpython-6/", None),
    "python": ("https://docs.python.org/3.10/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "numba": ("https://numba.readthedocs.io/en/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "lmfit": ("https://lmfit.github.io/lmfit-py/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "pyqtgraph": ("https://pyqtgraph.readthedocs.io/en/latest/", None),
    "arpes": ("https://arpes.readthedocs.io/en/latest/", None),
}


# plot configs
plot_formats = ["pdf"]
plot_basedir = "pyplots"
plot_html_show_formats = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ["_static"]
html_theme = "furo"
# html_theme_options = {
#     "repository_url": "https://github.com/kmnhan/erlabpy",
#     "use_repository_button": True,
#     "use_source_button": True,
# }

# -- LaTeX options -----------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#latex-options

latex_engine = "lualatex"
latex_table_style = ["booktabs"]
latex_elements = dict(
    fontpkg=r"""\usepackage{fontspec,unicode-math}
\setsansfont{Helvetica Neue}""",
)
