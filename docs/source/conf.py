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

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    # "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "matplotlib.sphinxext.plot_directive",
    # "IPython.sphinxext.ipython_directive",
    # "IPython.sphinxext.ipython_console_highlighting",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    #   'sphinx.ext.doctest',
    # "sphinx.ext.inheritance_diagram",
    "sphinx_qt_documentation",
]

templates_path = ["_templates"]
exclude_patterns = []


default_role = "obj"
add_function_parentheses = True
add_module_names = False
toc_object_entries = True
toc_object_entries_show_parents = "domain"

nitpicky = False
nitpick_ignore = [("py:class", "numpy.float64")]


# autosummary_generate = True

# autodoc_inherit_docstrings = False
# autodoc_typehints = "both"
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
napoleon_custom_sections = ["Signals"]


qt_documentation = "PySide6"
intersphinx_mapping = {
    "python": ("https://docs.python.org/3.10/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "numba": ("https://numba.readthedocs.io/en/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "pyqtgraph": ("https://pyqtgraph.readthedocs.io/en/latest/", None),
    "arpes": ("https://arpes.readthedocs.io/en/latest/", None),
}


# plot configs
plot_formats = ["pdf"]
plot_basedir = "pyplots"
plot_html_show_formats = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

# -- LaTeX options -----------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#latex-options

latex_engine = "lualatex"
latex_table_style = ["booktabs"]
latex_elements = dict(
    fontpkg=r"""\usepackage{fontspec,unicode-math}
\setsansfont{Helvetica Neue}""",
)
