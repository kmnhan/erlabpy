# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
# -- Imports -----------------------------------------------------------------

import importlib.metadata
import os

import pybtex.plugin
import pybtex.style.formatting
import pybtex.style.formatting.unsrt
import pybtex.style.template

# Build docs with PyQt6 since PySide6 is broken
# https://bugreports.qt.io/browse/PYSIDE-1884

os.environ["QT_API"] = "pyqt6"


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


project = "ERLab"
copyright = "2023, Kimoon Han"
author = "Kimoon Han"
release = importlib.metadata.version("erlab")
version = release


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    # "sphinx_autodoc_typehints",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "matplotlib.sphinxext.plot_directive",
    "matplotlib.sphinxext.figmpl_directive",
    # "IPython.sphinxext.ipython_directive",
    # "IPython.sphinxext.ipython_console_highlighting",
    # "sphinx.ext.inheritance_diagram",
    "sphinxcontrib.bibtex",
    "sphinx_qt_documentation",
    "sphinx_copybutton",
    "nbsphinx",
    "sphinx_design",
]


templates_path = ["_templates"]
exclude_patterns = []

default_role = "obj"

# nitpicky = False
# nitpick_ignore = [("py:class", "numpy.float64")]


# -- Autosummary and autodoc settings ----------------------------------------

autosummary_generate = True
autosummary_generate_overwrite = True

autodoc_inherit_docstrings = False
autodoc_typehints = "both"
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

# -- Napoleon settings -------------------------------------------------------

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

# -- nbsphinx options --------------------------------------------------------

nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
]

# -- Qt documentation & intersphinx ------------------------------------------

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
    "iminuit": ("https://scikit-hep.org/iminuit/", None),
}


# -- Plot configuration ------------------------------------------------------

plot_formats = ["pdf"]
plot_basedir = "pyplots"
plot_html_show_formats = False

# -- Misc. settings ----------------------------------------------------------

copybutton_exclude = ".linenos, .gp, .go"

# -- Bibliography settings ---------------------------------------------------

pybtex.style.formatting.unsrt.date = pybtex.style.template.words(sep="")[
    "(", pybtex.style.template.field("year"), ")"
]


class ApsStyle(pybtex.style.formatting.unsrt.Style):
    """
    APS style for BibTeX formatting, adapted from the conf.py file of the `mitiq
    library<https://github.com/unitaryfund/mitiq>`_.
    """

    def format_title(self, e, which_field, as_sentence=True):
        formatted_title = pybtex.style.template.field(
            which_field, apply_func=lambda text: text.capitalize()
        )
        formatted_title = pybtex.style.template.tag("em")[formatted_title]
        if as_sentence:
            return pybtex.style.template.sentence[formatted_title]
        else:
            return formatted_title

    def get_article_template(self, e):
        volume_and_pages = pybtex.style.template.first_of[
            # volume and pages
            pybtex.style.template.optional[
                pybtex.style.template.join[
                    " ",
                    pybtex.style.template.tag("strong")[
                        pybtex.style.template.field("volume")
                    ],
                    ", ",
                    pybtex.style.template.field(
                        "pages",
                        apply_func=pybtex.style.formatting.unsrt.dashify,
                    ),
                ],
            ],
            # pages only
            pybtex.style.template.words[
                "pages",
                pybtex.style.template.field(
                    "pages", apply_func=pybtex.style.formatting.unsrt.dashify
                ),
            ],
        ]
        template = pybtex.style.formatting.toplevel[
            self.format_names("author"),
            self.format_title(e, "title"),
            pybtex.style.template.sentence(sep=" ")[
                pybtex.style.template.field("journal"),
                pybtex.style.template.optional[volume_and_pages],
                pybtex.style.formatting.unsrt.date,
            ],
            self.format_web_refs(e),
        ]
        return template

    def get_book_template(self, e):
        template = pybtex.style.formatting.toplevel[
            self.format_author_or_editor(e),
            self.format_btitle(e, "title"),
            self.format_volume_and_series(e),
            pybtex.style.template.sentence(sep=" ")[
                pybtex.style.template.sentence(add_period=False)[
                    pybtex.style.template.field("publisher"),
                    pybtex.style.template.optional_field("address"),
                    self.format_edition(e),
                ],
                pybtex.style.formatting.unsrt.date,
            ],
            pybtex.style.template.optional[
                pybtex.style.template.sentence[self.format_isbn(e)]
            ],
            pybtex.style.template.sentence[
                pybtex.style.template.optional_field("note")
            ],
            self.format_web_refs(e),
        ]
        return template

    def get_incollection_template(self, e):
        template = pybtex.style.formatting.toplevel[
            pybtex.style.template.sentence[self.format_names("author")],
            self.format_title(e, "title"),
            pybtex.style.template.words[
                "In",
                pybtex.style.template.sentence[
                    pybtex.style.template.optional[
                        self.format_editor(e, as_sentence=False)
                    ],
                    self.format_btitle(e, "booktitle", as_sentence=False),
                    self.format_volume_and_series(e, as_sentence=False),
                    self.format_chapter_and_pages(e),
                ],
            ],
            pybtex.style.template.sentence(sep=" ")[
                pybtex.style.template.sentence(add_period=False)[
                    pybtex.style.template.optional_field("publisher"),
                    pybtex.style.template.optional_field("address"),
                    self.format_edition(e),
                ],
                pybtex.style.formatting.unsrt.date,
            ],
            self.format_web_refs(e),
        ]
        return template


pybtex.plugin.register_plugin("pybtex.style.formatting", "apsstyle", ApsStyle)

bibtex_bibfiles = ["refs.bib"]
bibtex_default_style = "apsstyle"
bibtex_footbibliography_header = ".. rubric:: References"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
pygments_dark_style = "monokai"
language = "en"

html_static_path = ["_static"]
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
    "light_css_variables": {
        "color-brand-primary": "#6d50bf",
        "color-brand-content": "#6d50bf",
    },
    "dark_css_variables": {
        "color-brand-primary": "#a180ff",
        "color-brand-content": "#a180ff",
    },
}

# -- LaTeX options -----------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#latex-options

latex_engine = "lualatex"
latex_table_style = ["booktabs"]
latex_elements = {
    "fontpkg": r"""\usepackage{fontspec,unicode-math}
""",
}
