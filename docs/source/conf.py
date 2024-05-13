# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
# -- Imports -----------------------------------------------------------------

import importlib.metadata
import inspect
import os
import sys

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
    "sphinx.ext.linkcode",
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

highlight_language = "python3"

# -- Linkcode settings -------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/linkcode.html


# based on numpy doc/source/conf.py
def linkcode_resolve(domain, info):
    """Determine the URL corresponding to Python object."""
    if domain != "py":
        return None

    modname = info["module"]
    fullname = info["fullname"]

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return None

    try:
        fn = inspect.getsourcefile(inspect.unwrap(obj))
    except TypeError:
        fn = None
    if not fn:
        return None

    try:
        source, lineno = inspect.getsourcelines(obj)
    except OSError:
        lineno = None

    if lineno:
        linespec = f"#L{lineno}-L{lineno + len(source) - 1}"
    else:
        linespec = ""

    import erlab

    fn = os.path.relpath(fn, start=os.path.dirname(erlab.__file__))

    return (
        f"https://github.com/kmnhan/erlabpy/blob/"
        f"v{version}/src/erlab/{fn}{linespec}"
    )


# -- Autosummary and autodoc settings ----------------------------------------

autosummary_generate = True
autosummary_generate_overwrite = True

autodoc_class_signature = "mixed"
autodoc_member_order = "groupwise"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": False,
}
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
# autodoc_type_aliases = {}
autodoc_typehints_format = "short"
autodoc_preserve_defaults = True
autodoc_inherit_docstrings = False


# -- Napoleon settings -------------------------------------------------------

napoleon_google_docstring = False
napoleon_numpy_docstring = True
# napoleon_include_init_with_doc = False
# napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
# napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = True
# napoleon_use_admonition_for_references = False
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_keyword = False
# napoleon_use_rtype = True
napoleon_preprocess_types = True

napoleon_type_aliases = {
    "np.float32": "float32",
    "numpy.float32": "float32",
    "np.float64": "float64",
    "numpy.float64": "float64",
    "xr.DataArray": "xarray.DataArray",
    "array-like": "`array-like <numpy.typing.ArrayLike>`",
    "array_like": "`array-like <numpy.typing.ArrayLike>`",
    "ColorType": "`ColorType <matplotlib.typing.ColorType>`",
    "RGBColorType": "`RGBColorType <matplotlib.typing.RGBColorType>`",
    "RGBAColorType": "`RGBAColorType <matplotlib.typing.RGBAColorType>`",
}
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
    "csaps": ("https://csaps.readthedocs.io/en/latest/", None),
    "iminuit": ("https://scikit-hep.org/iminuit/", None),
    "cmasher": ("https://cmasher.readthedocs.io/", None),
    "ipywidgets": ("https://ipywidgets.readthedocs.io/en/stable/", None),
    "joblib": ("https://joblib.readthedocs.io/en/latest/", None),
    "panel": ("https://panel.holoviz.org/", None),
    "hvplot": ("https://hvplot.holoviz.org/", None),
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


@pybtex.style.template.node
def top_level(children, data):
    return pybtex.style.template.sentence(sep=", ")[children].format_data(data)


class APSStyle(pybtex.style.formatting.unsrt.Style):
    """APS style for BibTeX formatting.

    Adapted from the conf.py file of the `mitiq
    library<https://github.com/unitaryfund/mitiq>`_.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.abbreviate_names = True

    def format_names(self, role, as_sentence=False):
        return super().format_names(role, as_sentence=as_sentence)

    def format_volume_and_series(self, e, as_sentence=False):
        return super().format_volume_and_series(e, as_sentence=as_sentence)

    def format_title(self, e, which_field, as_sentence=False):
        formatted_title = pybtex.style.template.field(
            which_field, apply_func=lambda text: text.capitalize()
        )

        formatted_title = pybtex.style.template.tag("em")[formatted_title]
        if as_sentence:
            return pybtex.style.template.sentence[formatted_title]
        else:
            return formatted_title

    def format_editor(self, e, as_sentence=False):
        editors = self.format_names("editor", as_sentence=False)
        if "editor" not in e.persons:
            return editors
        result = pybtex.style.template.join(sep=" ")["edited by", editors]
        if as_sentence:
            return pybtex.style.template.sentence[result]
        else:
            return result

    def format_address_organization_publisher_date(self, e, include_organization=True):
        if include_organization:
            organization = pybtex.style.template.optional_field("organization")
        else:
            organization = None
        return pybtex.style.template.first_of[
            pybtex.style.template.optional[
                pybtex.style.template.join(sep=", ")[
                    pybtex.style.template.sentence(add_period=False, sep=", ")[
                        organization, pybtex.style.template.optional_field("publisher")
                    ],
                    pybtex.style.template.join(sep=", ")[
                        pybtex.style.template.sentence(add_period=False)[
                            pybtex.style.template.optional_field("address")
                        ],
                        pybtex.style.template.field("year"),
                    ],
                ]
            ],
            pybtex.style.template.join(sep=", ")[
                organization,
                pybtex.style.template.optional_field("publisher"),
                pybtex.style.template.field("year"),
            ],
        ]

    def format_web_refs(self, e):
        return pybtex.style.template.sentence(add_period=False)[
            pybtex.style.template.optional[
                self.format_url(e),
                pybtex.style.template.optional[
                    " (visited on ", pybtex.style.template.field("urldate"), ")"
                ],
            ],
            pybtex.style.template.optional[self.format_eprint(e)],
            pybtex.style.template.optional[self.format_pubmed(e)],
            pybtex.style.template.optional[self.format_doi(e)],
        ]

    def format_journal_volume_page(self, e):
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
        journal_volume_page = pybtex.style.template.sentence(sep=" ", add_period=False)[
            pybtex.style.template.field("journal"),
            pybtex.style.template.optional[volume_and_pages],
            pybtex.style.formatting.unsrt.date,
        ]
        if "eprint" in e.fields:
            return pybtex.style.template.href[
                pybtex.style.template.join[
                    "https://arxiv.org/abs/",
                    pybtex.style.template.field("eprint", raw=True),
                ],
                journal_volume_page,
            ]
        elif "url" in e.fields:
            return pybtex.style.template.href[
                pybtex.style.template.field("url", raw=True), journal_volume_page
            ]
        elif "pubmed" in e.fields:
            return pybtex.style.template.href[
                pybtex.style.template.join[
                    "https://www.ncbi.nlm.nih.gov/pubmed/",
                    pybtex.style.template.field("pubmed", raw=True),
                ],
                journal_volume_page,
            ]
        elif "doi" in e.fields:
            return pybtex.style.template.href[
                pybtex.style.template.join[
                    "https://doi.org/", pybtex.style.template.field("doi", raw=True)
                ],
                journal_volume_page,
            ]
        else:
            return journal_volume_page

    def get_article_template(self, e):
        template = top_level[
            self.format_names("author"),
            self.format_title(e, "title"),
            self.format_journal_volume_page(e),
        ]
        return template

    def get_book_template(self, e):
        template = top_level[
            self.format_author_or_editor(e),
            self.format_btitle(e, "title"),
            self.format_volume_and_series(e),
            pybtex.style.template.sentence(sep=" ", add_period=True)[
                pybtex.style.template.sentence(add_period=False)[
                    pybtex.style.template.field("publisher"),
                    pybtex.style.template.optional_field("address"),
                    self.format_edition(e),
                ],
                pybtex.style.formatting.unsrt.date,
            ],
            pybtex.style.template.optional[
                pybtex.style.template.sentence(add_period=False)[self.format_isbn(e)]
            ],
            pybtex.style.template.sentence(add_period=False)[
                pybtex.style.template.optional_field("note")
            ],
            self.format_web_refs(e),
        ]
        return template

    def get_incollection_template(self, e):
        template = top_level[
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

    def get_inproceedings_template(self, e):
        template = top_level[
            pybtex.style.template.sentence[self.format_names("author")],
            self.format_title(e, "title"),
            pybtex.style.template.words[
                "in",
                pybtex.style.template.sentence(add_period=False)[
                    self.format_btitle(e, "booktitle", as_sentence=False),
                    # self.format_volume_and_series(e, as_sentence=False),
                    # self.format_chapter_and_pages(e),
                    pybtex.style.template.field("year"),
                    pybtex.style.template.sentence(add_period=False, sep=" ")[
                        pybtex.style.template.optional[
                            self.format_editor(e, as_sentence=False)
                        ],
                        pybtex.style.template.words(sep="")[
                            "(", self.format_address_organization_publisher_date(e), ")"
                        ],
                    ],
                    pybtex.style.template.join(sep=" ")[
                        "p.", pybtex.style.formatting.unsrt.pages
                    ],
                    pybtex.style.template.optional_field("note"),
                ],
            ],
            self.format_web_refs(e),
        ]
        return template


pybtex.plugin.register_plugin("pybtex.style.formatting", "apsstyle", APSStyle)

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
    # "light_css_variables": {
    #     "color-brand-primary": "#6d50bf",
    #     "color-brand-content": "#6d50bf",
    # },
    # "dark_css_variables": {
    #     "color-brand-primary": "#a180ff",
    #     "color-brand-content": "#a180ff",
    # },
    "source_repository": "https://github.com/kmnhan/erlabpy/",
    "source_branch": f"v{version}",
    "source_directory": "docs/source/",
}

# -- LaTeX options -----------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#latex-options

latex_engine = "lualatex"
latex_show_pagerefs = True
latex_table_style = ["booktabs", "colorrows"]
latex_elements = {
    "papersize": "a4paper",
    "pointsize": "10pt",
    "fontpkg": r"""\usepackage{fontspec,unicode-math}
\directlua{
    luaotfload.add_fallback("monofallback", {"DejaVuSansMono:mode=harf;", })
}
\setmainfont{Source Serif Pro}
\setsansfont{IBM Plex Sans}
\setmonofont{IBM Plex Mono}[Scale=0.9,RawFeature={fallback=monofallback}]
""",
}
