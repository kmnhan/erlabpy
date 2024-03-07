# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import erlab

project = "ERLabPy"
copyright = "2023, Kimoon Han"
author = "Kimoon Han"
release = erlab.__version__
version = erlab.__version__
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
    # "sphinx_autodoc_typehints",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "matplotlib.sphinxext.plot_directive",
    "matplotlib.sphinxext.figmpl_directive",
    # "IPython.sphinxext.ipython_directive",
    # "IPython.sphinxext.ipython_console_highlighting",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    #   'sphinx.ext.doctest',
    # "sphinx.ext.inheritance_diagram",
    "sphinx_qt_documentation",
    "sphinx_copybutton",
    "nbsphinx",
]

plot_srcset = ["2x"]

templates_path = ["_templates"]
exclude_patterns = []


default_role = "obj"
add_function_parentheses = True
add_module_names = True
toc_object_entries = True
toc_object_entries_show_parents = "domain"

# nitpicky = False
# nitpick_ignore = [("py:class", "numpy.float64")]


autosummary_generate = True
autosummary_generate_overwrite = True

autodoc_inherit_docstrings = False
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_class_signature = "mixed"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    # "exclude-members":("sigDataChanged",),
    "show-inheritance": False,
}
autodoc_typehints_description_target = "all"

# autodoc_type_aliases = {
# "numpy.float64": "float",
# "float64": "float",
# }

# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
# napoleon_include_init_with_doc = False
# napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
# napoleon_use_admonition_for_examples = False
# napoleon_use_admonition_for_notes = False
# napoleon_use_admonition_for_references = False
# napoleon_use_ivar = False
napoleon_use_param = True
# napoleon_use_rtype = True
napoleon_preprocess_types = True
# napoleon_type_aliases = {
#     "array-like": "ndarray <numpy.ndarray>",
#     "numpy.float64": "float",
#     "float64": "float",
# }
napoleon_attr_annotations = True
napoleon_custom_sections = [("Signals", "params_style")]

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
    "csaps": ("https://csaps.readthedocs.io/en/latest/", None),
}


# plot configs
plot_formats = ["pdf"]
plot_basedir = "pyplots"
plot_html_show_formats = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
language = "en"

html_static_path = ["_static"]
# html_theme_options = {
#     "repository_url": "https://github.com/kmnhan/erlabpy",
#     "use_repository_button": True,
#     "use_source_button": True,
# }
html_css_files = ["pied-piper-admonition.css"]
html_theme_options: dict[str, object] = {
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/kmnhan/erlabpy",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
    ],
}

# -- LaTeX options -----------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#latex-options

latex_engine = "lualatex"
latex_table_style = ["booktabs"]
latex_elements = dict(
    fontpkg=r"""\usepackage{fontspec,unicode-math}
\setsansfont{Helvetica Neue}""",
)
